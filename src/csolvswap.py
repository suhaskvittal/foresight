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

DEBUG = 1

R_SKIP_TH = 0.3

# TODO Folding function
# TODO Lookahead support

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
		current_layout = Layout.generate_trivial_layout(dag.qregs["q"])
		front_layer = dag.front_layer()
		finished = set()
		while front_layer:
			output_layers, next_layout = self._insert_swaps(current_layout, front_layer)
			for layer in output_layers:
				for node in layer:
					mapped_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
			for node in front_layer:
				finished.add(node)
			next_layer = []
			for node in front_layer:
				adj_list = dag.descendants(node)
				for next_node in adj_list:
					if all(x in finished for x in dag.ancestors(next_node) if x.type == 'op'):
						next_layer.append(next_node)
			front_layer = next_layer
			current_layout = next_layout
		# Apply measure.
		for p in self.coupling_map.physical_qubits:
			measure_op = DAGNode(
				type='op',
				op=Measure(),
				qargs=[current_layout[p]],
				cargs=[dag.cregs['c'][p]]
			)
			mapped_dag.apply_operation_back(measure_op.op, qargs=measure_op.qargs,\
											cargs=measure_op.cargs)
		return mapped_dag

	def _insert_swaps(self, current_layout, front_layer):
		output_layers = [[] for _ in range(3*self.max_swaps+1)]

		target_set = []
		for gnode in front_layer:
			if gnode.type != 'op' or gnode.name == 'measure':
				continue
			if len(gnode.qargs) != 2:
				output_layers[0].append(gnode)  # Not a 2-qubit operation.
			else:
				v0, v1 = gnode.qargs
				if self.coupling_map.graph.has_edge(current_layout[v0], current_layout[v1]):
					output_layers[0].append(gnode)
				else:
					target_set.append((v0, v1))			
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
		layer_used_sets = []

		best_soln_layers = []
		best_layer_used_sets = []
		best_soln_size = -1
		for _ in range(self.max_runs):
			soln_layers = []
			layer_used_sets = []
			soln_size = 0
			for i in range(self.max_swaps):
				curr_layer = []	
				used = set()  # Keep track of used vertices to avoid choosing bad edges.
				if i > 0:
					prev_used = layer_used_sets[-1]
				else:
					prev_used = None
				candidate_list = path_dag[i]
				candidate_list.sort(key=lambda x: x[1], reverse=True)
				for ((v, w), _) in candidate_list:
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
				soln_layers.append(curr_layer)
				layer_used_sets.append(used)
			if self._verify_swaps(soln_layers, target_set, current_layout)\
			and (soln_size < best_soln_size or best_soln_size == -1):	
				best_soln_layers = soln_layers
				best_layer_used_sets = layer_used_sets
				best_soln_size = soln_size
		soln_layers = best_soln_layers
		layer_used_sets = best_layer_used_sets
				
		# Finally, try to reduce the depth of the circuit by folding the layers.
		#soln_layers = self._clean_layers(self._fold_layers(soln_layers, layer_used_sets))
		soln_layers = self._clean_layers(soln_layers)
		# Add SWAPs to output_layers.
		new_layout = current_layout.copy()
		for i in range(len(soln_layers)):
			layer = soln_layers[i]
			for (p0, p1) in layer:
				# Add CNOT gates to output layer
				v0, v1 = new_layout[p0], new_layout[p1]
				cx_gate1 = DAGNode(
					type='op',
					op=CXGate(),
					qargs=[v0, v1]	
				)
				cx_gate2= DAGNode(
					type='op',
					op=CXGate(),
					qargs=[v1, v0]	
				)
				cx_gate3 = DAGNode(
					type='op',
					op=CXGate(),
					qargs=[v0, v1]	
				)
				output_layers[3*i].append(cx_gate1)
				output_layers[3*i+1].append(cx_gate2)
				output_layers[3*i+2].append(cx_gate3)
				# Apply sp1ap to modify running layout.
				new_layout[p0], new_layout[p1] = new_layout[p1], new_layout[p0]
		# Apply original operations.
		for (v0, v1) in target_set:
			cx_gate = DAGNode(
				type='op',
				op=CXGate(), 
				qargs=[v0, v1]
			)	
			output_layers[3*i+3].append(cx_gate)
		return output_layers, new_layout

	def _fold_layers(self, layers, layer_usage_list):	
		prev_folds = set()
		for i in range(1, self.max_swaps//2):
			layer = layers[-i]
			folds = set()
			removed_indices = []
			for (j, (v, w)) in enumerate(layer):
				if i > 0 and (v in prev_folds or w in prev_folds):
					continue  # Cannot fold this edge.
				if v not in layer_usage_list[i] and w not in layer_usage_list[i]:
					folds.add(v)
					folds.add(w)
					removed_indices.append(j)
			layers[-i] = [x for (j,x) in layers[-i] if j not in removed_indices] 
			prev_folds = folds
		return layers
	
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

	def _build_lp(self, current_layout, target_set, next_target_set=None):
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
			z = LpVariable(z_var, 0, 1)
			var_list = []
			for (j, (v, w)) in enumerate(conj):
				x_var = 'x%d' % n_xvar
				if (j, (v, w)) not in variable_mapping:
					x = LpVariable(x_var, 0, 1)
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
				if w == sink:
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

