"""
	author:	Suhas Vittal
	date:	14 September 2021 @ 2:44 p.m. EST
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from pulp import * 

import numpy as np

from copy import copy

DEBUG = 1

R_SKIP_TH = 0.5

def make_lp_var(name):
	return LpVariable(name, 0, 1)

class ConvexSolverSwap(TransformationPass):
	def __init__(self, coupling_map, seed=None, max_swaps=5, max_runs=1000, edge_weights=None):
		self.coupling_map = coupling_map		
		self.max_swaps = max_swaps
		self.max_runs = max_runs
		self.seed = seed

		self.requires = []
		self.preserves = []

	def run(self, dag):
		mapped_dag = dag._copy_circuit_metadata()
		canonical_register = dag.qregs["q"]
		current_layout = Layout.generate_trivial_layout(canonical_register)
		front_layer = dag.front_layer()
		finished = set()
		while front_layer:
			second_layer_set = set()
			for node in front_layer:
				for child in dag.descendants(node):
					second_layer_set.add(child)
			second_layer = list(second_layer_set)
				
			output_layers, next_layout = self._insert_swaps(current_layout, front_layer, canonical_register, next_layer=second_layer)
			for layer in output_layers:
				for node in layer:
					mapped_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
			for node in front_layer:
				finished.add(node)
			next_layer = []
			for node in front_layer:
				adj_list = dag.descendants(node)
				for next_node in adj_list:
					if all(x in finished for x in dag.ancestors(next_node) if x.type == 'op')\
					and next_node not in finished and next_node not in next_layer:
						next_layer.append(next_node)
			front_layer = next_layer
			current_layout = next_layout
		return mapped_dag

	def _insert_swaps(self, current_layout, front_layer, canonical_register, next_layer=None):
		if next_layer is None:
			next_layer = []

		output_layers = [[] for _ in range(self.max_swaps+2)]

		target_set = []
		next_target_set = []
		post_ops = []
		for gnode in front_layer:
			if gnode.type != 'op':
				continue
			if len(gnode.qargs) != 2 or gnode.name == 'measure':
				output_layers[0].append(self._remap_gate_for_layout(gnode, current_layout, canonical_register))
			else:
				v0, v1 = gnode.qargs
				if self.coupling_map.graph.has_edge(current_layout[v0], current_layout[v1]):
					output_layers[0].append(self._remap_gate_for_layout(gnode, current_layout, canonical_register))
				else:
					target_set.append((v0, v1))			
					post_ops.append(gnode)
		for gnode in next_layer:
			if gnode.type != 'op' or gnode.name == 'measure' or len(gnode.qargs) != 2:
				continue  # Do not update output layer
			else:
				v0, v1 = gnode.qargs
				next_target_set.append((v0, v1))
		if len(target_set) == 0:
			return [output_layers[0]], current_layout
		lp, var_mapping = self._build_lp(current_layout, target_set)
		status = lp.solve(GLPK(msg=False))
		if LpStatus[status] != 'Optimal':  # Terminate if we do not find an optimal solution.
			print('[ERROR] Status = %s' % LpStatus[status])
			exit()
		# Use the var_mapping dict to retrieve the values of the variables.
		# Represent the Path DAG (R) as a list of max_swaps layers.
		path_dag = [[] for _ in range(self.max_swaps)]
		for (i, e) in var_mapping:
			if value(var_mapping[(i,e)]) == 0.0:
				continue
			path_dag[i].append((e, value(var_mapping[(i,e)]))) 	
		# Greedily choose edges at each layer.
		soln_layers = []

		best_soln_layers = []
		best_soln_size = -1
		best_next_dist = -1
		for _ in range(self.max_runs):
			soln_layers = []
			layer_used_sets = []
			soln_size = 0
			swap_score_list = []
			for i in range(self.max_swaps):
				curr_layer = []	
				layer_score_list = []
				used = set()  # Keep track of used vertices to avoid choosing bad edges.
				if i > 0:
					prev_used = layer_used_sets[-1]
				else:
					prev_used = set()
				candidate_list = path_dag[i]
				candidate_list.sort(key=lambda x: x[1], reverse=True)
				for (j, ((v, w), s)) in enumerate(candidate_list):
					if np.random.random() <= R_SKIP_TH:
						continue
					if v in used or w in used:
						# Vertex is occupied.
						continue
					if i > 0 and not (v in prev_used or w in prev_used): 
						# This edge is a straggling edge.
						continue
					# If we are at this point, we will choose the highest scoring edge.
					curr_layer.append((v, w))
					soln_size += 1
					used.add(v)
					used.add(w)
					if v in prev_used:
						prev_used.remove(v)
					if w in prev_used:
						prev_used.remove(w)
					layer_score_list.append(j)
				if len(prev_used) > 0:
					# Remove hanging swaps.
					removed_swaps = []
					for (j, (v, w)) in enumerate(soln_layers[i-1]):
						if v in prev_used and w in prev_used:
							removed_swaps.append(j)
						soln_layers[i-1] = [x for (j,x) in enumerate(soln_layers[i-1]) if j not in removed_swaps]

				soln_layers.append(curr_layer)
				layer_used_sets.append(used)
				swap_score_list.append(layer_score_list)
			soln_layers, next_dist = self._fold_layers(soln_layers, current_layout,\
														next_target_set=next_target_set)
			soln_layers = self._clean_layers(soln_layers)
			score_modify = 0.5
			if self._verify_swaps(soln_layers, target_set, current_layout):
				if soln_size + next_dist < best_soln_size + best_next_dist\
				or best_soln_size == -1:
					best_soln_layers = soln_layers
					best_soln_size = soln_size
					best_next_dist = next_dist
					score_modify = 1
#				else:
#					if soln_size + next_dist == best_soln_size + best_next_dist\
#					and self._soln_hash_f(soln_layers) != self._soln_hash_f(best_soln_layers):
#						print('equally good candidate found,', self._soln_hash_f(soln_layers), self._soln_hash_f(best_soln_layers))
			# Modify scores of edges in path.
			for (i, score_layer) in enumerate(swap_score_list):
				for j in score_layer:
					(v, w), s = path_dag[i][j]
					path_dag[i][j] = (v, w), (score_modify*np.random.random()+score_modify)*s
		soln_layers = best_soln_layers
		if len(soln_layers) == 0:
			print('[ERROR] Empty solution.')
			exit()
				
		# Add SWAPs to output_layers.
		new_layout = current_layout.copy()
		for i in range(len(soln_layers)):
			layer = soln_layers[i]
			for (p0, p1) in layer:
				# Add CNOT gates to output layer
				v0, v1 = new_layout[p0], new_layout[p1]
				if (v0, v1) in target_set or (v1, v0) in target_set:
					continue
				swp_gate = DAGNode(
					type='op',
					op=SwapGate(),
					qargs=[v0, v1]	
				)
				output_layers[i+1].append(self._remap_gate_for_layout(swp_gate, new_layout, canonical_register))
				# Apply swap to modify running layout.
				new_layout[p0], new_layout[p1] = new_layout[p1], new_layout[p0]
		# Apply original operations.
		for gnode in post_ops:
			output_layers[i+2].append(self._remap_gate_for_layout(gnode, new_layout, canonical_register))
		return output_layers, new_layout
	
	def _remap_gate_for_layout(self, gnode, layout, canonical_register):
		new_gnode = copy(gnode)
		new_gnode.qargs = [canonical_register[layout[x]] for x in gnode.qargs]
		return new_gnode

	def _fold_layers(self, layers, current_layout, next_target_set=None):	
		if next_target_set is None or len(next_target_set) == 0:
			return layers, 0
		# Choose a fold such that we minimize the distance to the next targets.
		min_d_index = 0
		min_dist = -1
		for i in range(self.max_swaps):  # Compute point of minimum distance -- that will be the fold point.
			test_layout = current_layout.copy()
			# Perform reverse swaps
			j = 0
			while j < i:
				layer = layers[j]
				for (p0, p1) in layer:
					test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
				j += 1
			j = self.max_swaps - 1
			while j >= i:
				layer = layers[j]
				for (p0, p1) in layer:
					test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
				j -= 1
			# After test layout is completed, compute cumulative distance.
			dist = sum(self.coupling_map.distance_matrix[test_layout[v0], test_layout[v1]]\
							for (v0, v1) in next_target_set)
			if min_dist < 0 or dist < min_dist:
				min_dist = dist
				min_d_index = i
		# Once we compute the min_d_index, we fold the layers.
		if min_d_index == 0:
			return layers[::-1], min_dist  # Return reverse of current layers.
		elif min_d_index == self.max_swaps - 1:
			return layers, min_dist  # This is just the current order.
		else:
			left_tree = layers[:min_d_index]
			right_tree = layers[min_d_index:][::-1]
			folded_tree = []
			# We can be lazy and not fold :) Just do [left_tree, right_tree]
			folded_tree.extend(left_tree)
			folded_tree.extend(right_tree)
			return folded_tree, min_dist
		if next_target_set is None or len(next_target_set) == 0:
			return layers, 0
		# Choose a fold such that we minimize the distance to the next targets.
		min_d_index = 0
		min_dist = -1
		for i in range(self.max_swaps):  # Compute point of minimum distance -- that will be the fold point.
			test_layout = current_layout.copy()
			# Perform reverse swaps
			j = 0
			while j < i:
				layer = layers[j]
				for (p0, p1) in layer:
					test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
				j += 1
			j = self.max_swaps - 1
			while j >= i:
				layer = layers[j]
				for (p0, p1) in layer:
					test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
				j -= 1
			# After test layout is completed, compute cumulative distance.
			dist = sum(self.coupling_map.distance_matrix[test_layout[v0], test_layout[v1]]\
							for (v0, v1) in next_target_set)
			if min_dist < 0 or dist < min_dist:
				min_dist = dist
				min_d_index = i
		# Once we compute the min_d_index, we fold the layers.
		if min_d_index == 0:
			return layers[::-1], min_dist  # Return reverse of current layers.
		elif min_d_index == self.max_swaps - 1:
			return layers, min_dist  # This is just the current order.
		else:
			left_tree = layers[:min_d_index]
			right_tree = layers[min_d_index:][::-1]
			folded_tree = []
			# We can be lazy and not fold :) Just do [left_tree, right_tree]
			folded_tree.extend(left_tree)
			folded_tree.extend(right_tree)
			return folded_tree, min_dist

	def _clean_layers(self, layers):
		removed_layers = []
		for i in range(self.max_swaps):
			ii = -(i+1)
			removed_indices = []
			for (j, (v, w)) in enumerate(layers[ii]):
				if v == w:
					removed_indices.append(j)
			layers[ii] = [x for (j,x) in enumerate(layers[ii]) if j not in removed_indices]
			if len(layers[ii]) == 0:
				removed_layers.append(self.max_swaps + ii)
		layers = [x for (i,x) in enumerate(layers) if i not in removed_layers]
		return layers

	def _verify_swaps(self, cand_soln_layers, target_set, current_layout):
		test_layout = current_layout.copy()
		target_set_indicator = {x: 0 for x in target_set}
		for layer in cand_soln_layers:
			for (p0, p1) in layer:
				test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
		for p0 in self.coupling_map.physical_qubits:
			v0 = test_layout[p0]
			for p1 in self.coupling_map.neighbors(p0):
				v1 = test_layout[p1]
				if (v0, v1) in target_set:
					target_set_indicator[(v0, v1)] = 1
				elif (v1, v0) in target_set:
					target_set_indicator[(v1, v0)] = 1
		return all(target_set_indicator[x] == 1 for x in target_set)

	def _build_lp(self, current_layout, target_set):
		conjunctions = []	
		for (v0, v1) in target_set:
			p0, p1 = current_layout[v0], current_layout[v1]
			# Find paths of length at most max_swaps from p0 to p1
			path_list = self._bfs(p0, p1)
			conjunctions.extend(path_list)
		# Initialize LP
		lp = LpProblem('Solver', LpMinimize)
		# Build LP Variables
		variable_mapping = dict()
		n_xvar = 0
		edge_var_list = []
		z_var_list = []
		for (i, conj) in enumerate(conjunctions):
			z_var = 'z%d' % i
			z = make_lp_var(z_var)
			var_list = []
			for (j, (v, w)) in enumerate(conj):
				x_var = 'x%d' % n_xvar
				if (j, (v, w)) not in variable_mapping:
					x = make_lp_var(x_var)
					variable_mapping[(j, (v, w))] = x 
					edge_var_list.append(x)
					n_xvar += 1
				else:
					x = variable_mapping[(j, (v, w))]
				lp += (x + (1.0 - z) >= 1.0)
				var_list.append(1.0 - x)
			lp += (lpSum(var_list) + z >= 1.0)	
			z_var_list.append(z)
		lp += (lpSum(z_var_list) >= 1.0)
		# Build objective
		lp += lpSum(edge_var_list)
		return lp, variable_mapping
		
	def _bfs(self, source, sink, depth=None):
		if depth is None:
			depth = self.max_swaps

		path_list = []
		queue = [(source, [])]
		while len(queue) > 0:
			v, prev = queue.pop(0) 
			if len(prev) == depth:
				continue
			for w in self.coupling_map.neighbors(v):  # Add adjacent vertices.
				prev_cpy = prev.copy()
				if len(prev) > 0 and (prev[-1] == (v, w) or prev[-1] == (w, v)):
					continue
				if depth - len(prev_cpy) < self.coupling_map.distance_matrix[w, sink]:  # too far.
					continue
				if w == sink:
					prev_cpy.append((v, w))
					while len(prev_cpy) < depth:
						prev_cpy.append((w, w))
					path_list.append(prev_cpy)
					continue
				if v < w:
					prev_cpy.append((v, w))
				else:
					prev_cpy.append((w, v))
				queue.append((w, prev_cpy))
		return path_list

	def _soln_hash_f(self, soln):
		h = 0
		PRIME = 766453 
		for layer in soln:
			for (p0, p1) in layer:
				h += ((2**p0)*(3**p1)) % PRIME
		return h	
		
