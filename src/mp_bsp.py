"""
	author: Suhas Vittal
	date:	20 October 2021
"""

from qiskit.circuit import Qubit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from mp_swap_priority_queue import SwapPriorityQueue
from mp_swap_tree import SwapTreeNode
from mp_dist import _floyd_warshall

import numpy as np

from copy import copy, deepcopy
from collections import deque

# MPATH_BSP - BSP: Best Swap has Priority.
class MPATH_BSP(TransformationPass):
	def __init__(self, coupling_map, tree_width_limit=64, edge_weights=None):
		super().__init__()

		self.coupling_map = coupling_map		
		self.tree_width_limit = tree_width_limit
		self.distance_matrix, _ = _floyd_warshall(coupling_map, edge_weights=edge_weights)

		self.fake_run = False
			
	def run(self, dag, primary_layer_view=None, secondary_layer_view=None):
		canonical_register = dag.qregs["q"]
		current_layout = Layout.generate_trivial_layout(canonical_register)
		
		# Try to see what is executable immediately.
		marked_node_map = {}
		for (_, input_node) in dag.input_map.items():
			for (_, node, edge_data) in dag.edges(input_node):
				if node.type == 'op' and isinstance(edge_data, Qubit):
					if node not in marked_node_map:
						marked_node_map[node] = 0
					marked_node_map[node] += 1
		execution_root = SwapTreeNode(None, [], current_layout, [], marked_node_map)
		exec_deque = deque(dag.front_layer())
		self._traverse_execution_deque(exec_deque, execution_root, dag, canonical_register)
		execution_root.last_modifying_ancestor = execution_root
		execution_root.is_lma = True

		current_tree_layer = [execution_root]
		lma_list = [execution_root]
		min_dag = None
		while current_tree_layer:
			if len(current_tree_layer) > self.tree_width_limit:
				lma_list = self._contract_tree(lma_list, dag, canonical_register)	
				if len(lma_list) > self.tree_width_limit:
					lma_list = list(np.random.choice(lma_list, size=self.tree_width_limit, replace=False))
				current_tree_layer = copy(lma_list)
			next_tree_layer = []
			for tree_node in current_tree_layer:
				if tree_node.front_layer:
					children, invalids = self.exec_on_subtree(tree_node, dag, canonical_register)
					for (i, c) in enumerate(children):
						next_tree_layer.append(c)
						if c.is_lma:
							lma_list.append(c)
				else:
					# Apply output list and exit.
					mapped_dag = dag._copy_circuit_metadata()
					final_layout = current_layout
					for node in tree_node.output_list:
						if not self.fake_run:
							mapped_dag.apply_operation_back(node.op, node.qargs, node.cargs)
						final_layout = tree_node.current_layout
					if min_dag is None or mapped_dag.size() < min_dag.size() or (mapped_dag.size() == min_dag.size() and mapped_dag.depth() < min_dag.depth()):
						min_dag = mapped_dag
						self.property_set['final_layout'] = final_layout
			current_tree_layer = next_tree_layer
		return min_dag
	
	def exec_on_subtree(self, root, dag, canonical_register):
		# Build post layers
		post_layers = self._get_post_layers(root, dag)

		swap_candidates = []
		seen = set()
		current_layout = root.current_layout
		for node in root.front_layer:
			v0, v1 = node.qargs  # These are all two qubit operations.
			p0, p1 = current_layout[v0], current_layout[v1]
			for p in [p0, p1]:
				for q in self.coupling_map.neighbors(p):
					score = self._score((p, q), root.front_layer, post_layers, current_layout)
					swap_candidates.append(((p, q), score))
		swap_queue = SwapPriorityQueue.buildheap(swap_candidates)
		min_swap, min_score = swap_queue.dequeue()
		min_swap_list = [min_swap]
		while swap_queue.size > 0:
			swap, score = swap_queue.dequeue()
			if score > min_score*1:
				break
			min_swap_list.append(swap)
		# Create SwapTreeNodes for all swaps in min_swap_list
		child_array = []
		invalidated = []
		for (p0, p1) in min_swap_list:
			test_layout = root.current_layout.copy()
			test_layout.swap(p0, p1)
			child_tree_node = SwapTreeNode((p0, p1), [], test_layout, copy(root.output_list), copy(root.marked_node_map))
			root.children.append(child_tree_node)
			child_tree_node.parent = root
			child_tree_node.last_modifying_ancestor = root.last_modifying_ancestor
			# Check if anything in the front layer has been satisfied.
			exec_deque = deque([])
			modified = False
			for node in root.front_layer:
				q0, q1 = node.qargs
				if self.coupling_map.graph.has_edge(test_layout[q0], test_layout[q1]):
					exec_deque.append(node)
					modified = True
				else:
					child_tree_node.front_layer.append(node)  # Still needs to be completed.
			if modified:
				# The child node has modified the dag by completing ops.
				self._collapse_swap_tree(child_tree_node, canonical_register)
				# Unreference parent.
				child_tree_node.parent = None
				child_tree_node.last_modifying_ancestor = child_tree_node  
				child_tree_node.is_lma = True
				self._traverse_execution_deque(exec_deque, child_tree_node, dag, canonical_register)
				invalidated.append(root.last_modifying_ancestor)
			child_array.append(child_tree_node)
		return child_array, invalidated
	
	def _contract_tree(self, lma_list, dag, canonical_register):
		# Contract for each last modifying ancestor.
		# Last modifying ancestor is contiguous.
		# Perform a DFS on each lma if they are valid.
		new_lma_list = []
		for lma in lma_list:
			if lma.valid == 0:
				continue
			min_leaves, min_score = None, -1
			output_size_ref = None
			dfs_stack = [(lma, [])]
			while dfs_stack:
				tree_node, swap_list = dfs_stack.pop()
				swap_list_cpy = copy(swap_list)
				if tree_node.swap:
					p0, p1 = tree_node.swap
					swap_op = DAGNode(op=SwapGate(), qargs=[tree_node.current_layout[p0], tree_node.current_layout[p1]], type='op')
					swap_list_cpy.append(self._remap_gate_for_layout(swap_op, tree_node.current_layout, canonical_register))
				if tree_node.children:
					for c in tree_node.children:
						if c.is_lma:
							continue
						dfs_stack.append((c, swap_list_cpy))
				else:  # This is a leaf.
					# Score the leaf.
					post_layers = self._get_post_layers(tree_node, dag)
					if output_size_ref is None:
						output_size_ref = len(swap_list_cpy) + len(tree_node.output_list)
					score = self._distf(tree_node.front_layer, post_layers, tree_node.current_layout)\
						+ (len(swap_list_cpy) + len(tree_node.output_list))/output_size_ref * 0.0
					# update min leaves
					if min_score < 0 or score <= min_score:
						for node in swap_list_cpy:
							tree_node.output_list.append(node)
						tree_node.last_modifying_ancestor = tree_node
						tree_node.is_lma = True
						tree_node.swap = None
						if min_score < 0 or score < min_score:
							min_leaves = [tree_node]
							min_score = score
						elif min_score == score:
							min_leaves.append(tree_node)
			if min_leaves:
				new_lma_list.extend(min_leaves)
		return new_lma_list

	def _collapse_swap_tree(self, tree_node, canonical_register):
		# Get swaps until we reach last modifying ancestor (if lma, then valid bit of node is 0).
		tree_node.last_modifying_ancestor.valid = 0  # Easiest way to check, change back later.
		swap_list = []
		curr = tree_node
		while curr.valid == 1:
			swap_list.append(curr.swap)
			curr = curr.parent
		tree_node.last_modifying_ancestor.valid = 1
		# Push swaps onto output list
		tree_node.swap = None
		while swap_list:
			p0, p1 = swap_list.pop()
			swap_op = DAGNode(op=SwapGate(), qargs=[tree_node.current_layout[p0], tree_node.current_layout[p1]], type='op')
			tree_node.output_list.append(self._remap_gate_for_layout(swap_op, tree_node.current_layout, canonical_register))

	def _traverse_execution_deque(self, exec_deque, tree_node, dag, canonical_register):
		current_layout = tree_node.current_layout
		while exec_deque:
			node = exec_deque.popleft()
			if len(node.qargs) == 2:
				q0, q1 = node.qargs
				if self.coupling_map.graph.has_edge(current_layout[q0], current_layout[q1]):
					tree_node.output_list.append(self._remap_gate_for_layout(node, current_layout, canonical_register))  # Place in execution list.
				else:
					tree_node.front_layer.append(node)
					continue
			else:
				tree_node.output_list.append(self._remap_gate_for_layout(node, current_layout, canonical_register))  # Place in execution list.
			for (_, child, edge_data) in dag.edges(node):
				if child.type != 'op' or not isinstance(edge_data, Qubit):
					continue
				if child not in tree_node.marked_node_map:
					tree_node.marked_node_map[child] = 0
				tree_node.marked_node_map[child] += 1
				if tree_node.marked_node_map[child] == len(child.qargs):
					# Then all dependencies are satisified, add to exec deque.	
					exec_deque.append(child)
	
	def _get_post_layers(self, tree_node, dag, number=20):
		post_layers = [[]]

		mnm = tree_node.marked_node_map.copy()
		n = 0
		curr_layer = tree_node.front_layer
		while curr_layer:
			post = []
			next_layer = []
			for node in curr_layer:
				for (_, child, edge_data) in dag.edges(node):
					if child.type != 'op' or not isinstance(edge_data, Qubit):
						continue
					if child not in mnm:
						mnm[child] = 0
					mnm[child] += 1
					if mnm[child] == len(child.qargs):
						next_layer.append(child)
						if len(child.qargs) == 2:
							post.append(child)
							n += 1
					if n >= number:
						break
				if n >= number:
					break
			if n >= number:
				break
			curr_layer = next_layer
			if post:
				post_layers.append(post)
		return post_layers	

	def _remap_gate_for_layout(self, op, layout, canonical_register):
		new_op = copy(op)
		new_op.qargs = [canonical_register[layout[x]] for x in op.qargs]
		return new_op
	
	def _score(self, proposed_swap, front_layer, post_layers, current_layout):
		p0, p1 = proposed_swap
		test_layout = current_layout.copy()
		test_layout.swap(p0, p1)
		return self._distf(front_layer, post_layers, test_layout)

	def _distf(self, front_layer, post_layers, current_layout):
		if len(front_layer) == 0:
			return 0
		dist = 0.0
		for node in front_layer:
			q0, q1 = node.qargs
			dist += self.distance_matrix[current_layout[q0]][current_layout[q1]]
		dist = dist / len(front_layer)
		num_ops = 0
		post_dist = 0.0
		for (r, layer) in enumerate(post_layers):
			if not layer:
				continue
			sub_sum = 0.0
			for node in layer:
				q0, q1 = node.qargs
				num_ops += 1
				sub_sum += self.distance_matrix[current_layout[q0]][current_layout[q1]]
			post_dist += sub_sum
		if num_ops > 0:
			dist += post_dist*0.5/num_ops
		return dist

