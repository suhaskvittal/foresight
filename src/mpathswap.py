"""
	author: Suhas Vittal
	date:	29 September 2021 @ 2:09 p.m. EST
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

import numpy as np

from copy import copy, deepcopy

from ppc import PriorityPathCollection
from dstruct import SumTreeNode

MPSWAP_ERRNO = 0

class MultipathSwap(TransformationPass):
	def __init__(self, coupling_map, seed=None, max_swaps=5, max_path_limit=128, max_lookahead=16, solution_cap=4, edge_weights=None):
		super().__init__()

		self.coupling_map = coupling_map		
		self.max_swaps = max_swaps
		self.max_path_limit = max_path_limit
		self.max_lookahead = max_lookahead
		self.solution_cap = solution_cap
		self.seed = seed

		self.path_memoizer = {}
			
	def run(self, dag, primary_layer_view=None, secondary_layer_view=None):
		MPSWAP_ERRNO = 0  # reset ERRNO

		mapped_dag = dag._copy_circuit_metadata()
		canonical_register = dag.qregs["q"]
		current_layout = Layout.generate_trivial_layout(canonical_register)
		
		if primary_layer_view is None or secondary_layer_view is None:
			primary_layer_view, secondary_layer_view = self.property_set['primary_layer_view'], self.property_set['secondary_layer_view']
		solutions = self.deep_solve(primary_layer_view, secondary_layer_view, [(current_layout, mapped_dag)], canonical_register)

		if len(solutions) == 0:
			MPSWAP_ERRNO = 1  # error has occurred.
			return mapped_dag

		_, mapped_dag = solutions[0]  # Just get first solution, doesn't really matter.
		return mapped_dag
	
	def deep_solve(self, primary_layer_view, secondary_layer_view, base_solver_queue, canonical_register):
		if len(primary_layer_view) == 0:
			return base_solver_queue

		# Build tree of output layers.
		solver_queue = []
		leaves = []
		for (i, (base_layout, base_dag)) in enumerate(base_solver_queue):
			root_node = SumTreeNode(base_dag, 0, None, [])  # Other nodes won't keep a DAG.
			leaves.append(root_node)
			solver_queue.append((base_layout, i))
		look = 0
		while look < self.max_lookahead and len(solver_queue) < self.solution_cap:
			if len(primary_layer_view) == 0:
				break
			next_solver_queue = []
			next_leaves = []
			for (current_layout, parent_id) in solver_queue:  # Empty out queue into next_solver_queue.
				parent = leaves[parent_id]
				# Find parent corresponding to parent id.
				solutions, minus_look = self.shallow_solve(primary_layer_view, secondary_layer_view, current_layout, canonical_register)
				# Apply solutions non-deterministically to current_dag.
				for (i, (output_layers, new_layout)) in enumerate(solutions):
					# Create node for each candidate solution.
					node = SumTreeNode(output_layers, parent.sum_data + sum(len(layer) for layer in output_layers), parent, []) 
					parent.children.append(node)
					next_leaves.append(node)
					next_solver_queue.append((new_layout, len(next_leaves) - 1))
			solver_queue = next_solver_queue
			leaves = next_leaves
			primary_layer_view.pop(0)
			secondary_layer_view.pop(0)
			look += 1
		# Now, we simply check the leaves of the output layer tree. We select the leaf with the minimum sum.
		min_leaves = []
		min_sum = -1
		for (i, leaf) in enumerate(leaves):
			layout, _ = solver_queue[i]
			if min_sum < 0 or leaf.sum_data < min_sum:
				min_leaves = [(leaf, layout)]
				min_sum = leaf.sum_data
			elif leaf.sum_data == min_sum:
				min_leaves.append((leaf, layout))
		# Now that we know the best leaves, we simply just need to build the corresponding dags.
		# Traverse up the leaves -- there is only one path to a root.
		min_leaves = min_leaves[:self.solution_cap]
		min_solutions = []
		for (leaf, layout) in min_leaves:
			output_layer_collection = []
			curr = leaf
			while curr.parent != None:
				output_layer_collection.append(curr.obj_data)
				curr = curr.parent
			# Now we should be at the root, copy the DAG in the root's obj_data.
			mapped_dag = deepcopy(curr.obj_data)
			for i in range(len(output_layer_collection)):
				# Apply in reverse order of the collection (as collection was built in reverse order).
				output_layers = output_layer_collection[-(i+1)]
				for layer in output_layers:
					for node in layer:
						mapped_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
			min_solutions.append((layout, mapped_dag))

		return self.deep_solve(primary_layer_view, secondary_layer_view, min_solutions, canonical_register)

	def shallow_solve(self, primary_layer_view, secondary_layer_view, current_layout, canonical_register):
		path_collection_list = []	

		front_layer = primary_layer_view[0]
		output_layers = [[] for _ in range(2*self.max_swaps+2)]
		for op in secondary_layer_view[0]:
			output_layers[0].append(self._remap_gate_for_layout(op, current_layout, canonical_register))

		if len(primary_layer_view) == 1:
			post_primary_layer_view = []
		else:
			post_primary_layer_view = primary_layer_view[1:]
		
		target_list = []
		post_ops = []
		target_to_op = {}
		for op in front_layer:
			q0, q1 = op.qargs
			# Filter out all operations that are currently adjacent.
			if self.coupling_map.graph.has_edge(current_layout[q0], current_layout[q1]):
				output_layers[0].append(self._remap_gate_for_layout(op, current_layout, canonical_register))
			else:  # Otherwise, get path candidates and place it in path_collection_list.
				path_collection_list.append(self.path_find_and_fold(q0, q1, post_primary_layer_view, current_layout))
				post_ops.append(op)
				target_list.append((q0, q1))
				target_to_op[(q0, q1)] = op
		# Build PPC and get candidate list.
		if len(path_collection_list) == 0:
			return [(output_layers, current_layout)], 0
#		print('\t', [(current_layout[q0], current_layout[q1]) for (q0, q1) in target_list])
		ppc = PriorityPathCollection(path_collection_list, len(self.coupling_map.physical_qubits), len(path_collection_list))
		candidate_list, suggestions = ppc.find_and_join(self, target_list, current_layout, post_primary_layer_view)
		if candidate_list is None:  # We failed, take the suggestions.
			for index_list in suggestions:
				if len(index_list) == 0:
					continue
				target_sub_list = [target_list[i] for i in index_list]
				# Create new primary layer and secondary layer.
				primary_layer_view.insert(1, [target_to_op[target] for target in target_sub_list]) 
				secondary_layer_view.insert(1, [])
			return [(output_layers, current_layout)], len(suggestions)
		# Compute all solutions.
		solutions = []
		
		for soln in candidate_list:
#			print(soln)
			output_layers_cpy = deepcopy(output_layers)
			new_layout = current_layout.copy()
			for (i, layer) in enumerate(soln):  # Perform the requisite swaps.
				for (p0, p1) in layer:
					if p0 == p1:
						continue
					v0, v1 = new_layout[p0], new_layout[p1]
					swp_gate = DAGNode(
						type='op',
						op=SwapGate(),
						qargs=[v0, v1]	
					)
					if i + 1 >= len(output_layers_cpy):
						output_layers_cpy.append([])
					output_layers_cpy[i+1].append(self._remap_gate_for_layout(swp_gate, new_layout, canonical_register))
					# Apply swap to modify running layout.
					new_layout[p0], new_layout[p1] = new_layout[p1], new_layout[p0]
			# Apply the operations after completing all the swaps.
			for op in post_ops:  
				if i + 2 >= len(output_layers_cpy):
					output_layers_cpy.append([])
				output_layers_cpy[i+2].append(self._remap_gate_for_layout(op, new_layout, canonical_register))
			solutions.append((output_layers_cpy, new_layout))  # Save layers to apply to DAG and corresponding layout.
		return solutions, 0

	def path_find_and_fold(self, v0, v1, post_primary_layer_view, current_layout):
		p0, p1 = current_layout[v0], current_layout[v1]
		if self.coupling_map.graph.has_edge(p0, p1):
			return []
		# Get path candidates from _path_select.
		path_candidates = self._path_select(p0, p1)
		# Now, we must fold the candidates and choose their minfolds.
		return [self._path_minfold(c, current_layout, post_primary_layer_view) for c in path_candidates]
	
	def _path_select(self, source, sink):
		memoizer_entry = (source, sink) if source < sink else (sink, source)
		if memoizer_entry in self.path_memoizer:
			return self.path_memoizer[memoizer_entry]
	
		path_candidates = []

		bfs_queue = [(source, [])]
		# Perform modified DFS to find paths upto length "max_swaps".
		# We will not mark vertices as visited.
		while bfs_queue and len(path_candidates) < self.max_path_limit:
			v, prev = bfs_queue.pop(0)	
			if v == sink:
				path_candidates.append(prev)  # We have found the sink, add the corresponding path.
				continue
			for w in self.coupling_map.graph.neighbors(v):
				# IF w is too far from the sink, THEN do not add w.
				edge = (v, w) if v < w else (w, v)	
				if len(prev) + self.coupling_map.distance_matrix[w, sink] > self.max_swaps:
					continue
				elif len(prev) > 0 and prev[-1] == edge:
					continue
				prev_cpy = copy(prev)
				prev_cpy.append(edge)
				bfs_queue.append((w, prev_cpy))
			# Also add a self-edge.
#			if len(prev) + self.coupling_map.distance_matrix[v, sink] <= self.max_swaps:
#				prev.append((v, v))
#				bfs_queue.append((v, prev))
	
		self.path_memoizer[memoizer_entry] = path_candidates
		return path_candidates
	
	def _path_minfold(self, path, current_layout, post_primary_layer_view):
		min_dist = -1
		min_fold = []
		latest_layout = current_layout
		latest_fold = None
		for i in range(len(path)):
			fold = [] if latest_fold is None else copy(latest_fold)
			test_layout = latest_layout.copy()
			# Perform all swaps except path[i].
			# If i = 0, then perform fold that eliminates first swap.
			if i == 0:
				j = len(path) - 1
				while j > i:
					p0, p1 = path[j]
					fold.append(path[j])
					test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
					j -= 1
			else:  # if i != 0, then undo last backwards swap in last fold, and perform forwards swap prior to folded swap. 
				f0, f1 = path[i-1]
				b0, b1 = path[i]
				test_layout[b0], test_layout[b1] = test_layout[b1], test_layout[b0]  # Flip them back.
				test_layout[f0], test_layout[f1] = test_layout[f1], test_layout[f0]  # Perform forward flip.
				fold.pop()  # Remove last swap.
				fold.insert(i-1, path[i-1])  # Insert new swap at beginning.
			# Update latest layout.
			latest_layout = test_layout
			latest_fold = fold
			# Compute distance to post primary layer.
			dist = self._distf(post_primary_layer_view, test_layout)
			if dist < min_dist or min_dist == -1:
				min_dist = dist
				min_fold = fold
		return min_fold, min_dist + len(path)

	def _remap_gate_for_layout(self, op, layout, canonical_register):
		new_op = copy(op)
		new_op.qargs = [canonical_register[layout[x]] for x in op.qargs]
		return new_op

	def _verify_and_measure(self, soln, target_list, current_layout, post_primary_layer_view, verify_only=False):
		test_layout = current_layout.copy()
		size = 0
		for layer in soln:
			for (p0, p1) in layer:
				test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
				size += 1
		max_allowed_size = 0
		for (v0, v1) in target_list:
			max_allowed_size += self.coupling_map.distance_matrix[current_layout[v0], current_layout[v1]] - 1
			p0, p1 = test_layout[v0], test_layout[v1]
			if self.coupling_map.distance_matrix[p0, p1] > 1:
				return False, 0
		if size > max_allowed_size:
			return False, 0
		# If we have gotten to this point, then our soln is good.
		if verify_only:
			dist = 0
		else:
			dist = self._distf(post_primary_layer_view, test_layout)

		return True, dist + size

	def _distf(self, post_primary_layer_view, test_layout):
		dist = 0.0
		for r in range(0, min(len(post_primary_layer_view), 100+1)):
			post_layer = post_primary_layer_view[r]
			sub_sum = 0.0
			for op in post_layer:
				v0, v1 = op.qargs
				sub_sum += self.coupling_map.distance_matrix[test_layout[v0], test_layout[v1]]
			dist += sub_sum / ((r+1))
		return dist

