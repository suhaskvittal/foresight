"""
	author: Suhas Vittal
	date:	29 September 2021 @ 2:09 p.m. EST
"""

from qiskit.circuit import Qubit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

import numpy as np

from copy import copy, deepcopy
from collections import deque

class SwapTreeNode:
	def __init__(self, swap, front_layer, current_layout, output_list, marked_node_map):
		self.swap = swap
		self.front_layer = front_layer
		self.current_layout = current_layout
		self.output_list = output_list
		self.parent = None
		self.last_modifying_ancestor = None
		self.is_lma = False
		self.children = []
		self.valid = 1
		self.marked_node_map = marked_node_map

class SwapPriorityQueue:
	def __init__(self):
		self.backing_array = [None]
		self.size = 0
	
	def buildheap(array):
		pq = SwapPriorityQueue()	
		pq.backing_array.extend(array)
		pq.size = len(array)

		i = len(pq.backing_array) >> 1
		while i >= 1:
			pq._downheap(i)
			i -= 1
		return pq
	
	def enqueue(self, x, score):
		self.backing_array.append((x, score))
		self.size += 1
		self._upheap(self.size)
	
	def dequeue(self):
		if self.size == 0:
			return None
		root = self.backing_array[1]
		self.size -= 1
		if self.size > 0:
			self.backing_array[1] = self.backing_array.pop()
			self._downheap(1)
		return root
	
	def head(self):
		return self.backing_array[1]

	def _upheap(self, from_index):
		x, score = self.backing_array[from_index]
		curr_index = from_index
		while curr_index > 1:
			parent_index = curr_index >> 1  # parent is index/2
			px, pscore = self.backing_array[parent_index]
			if pscore < score:  # Then we are done.
				break
			# Otherwise, swap parent and child.
			self.backing_array[curr_index], self.backing_array[parent_index] =\
				self.backing_array[parent_index], self.backing_array[curr_index]
			curr_index = parent_index
	
	def _downheap(self, from_index):
		x, score = self.backing_array[from_index]
		curr_index = from_index
		while curr_index < self.size + 1:
			left_index = curr_index << 1
			right_index = left_index + 1
			mx, mscore = None, -1
			min_index = -1
			if left_index < self.size + 1:
				lx, lscore = self.backing_array[left_index]
				mx, mscore = lx, lscore
				min_index = left_index
			if right_index < self.size + 1:
				rx, rscore = self.backing_array[right_index]
				if mscore < 0 or rscore < mscore:  
					mx, mscore = rx, rscore
					min_index = right_index
			if mscore > score or mx is None: 
				break
			self.backing_array[curr_index], self.backing_array[min_index] =\
				self.backing_array[min_index], self.backing_array[curr_index]
			curr_index = min_index
			
class MultipathSwap(TransformationPass):
	def __init__(self, coupling_map, seed=None, max_swaps=5, max_path_limit=128, max_lookahead=16, solution_cap=4, edge_weights=None):
		super().__init__()

		self.coupling_map = coupling_map		
		self.slack = 2
		self.max_swaps = max_swaps
		self.max_path_limit = max_path_limit
		self.max_lookahead = max_lookahead
		self.solution_cap = solution_cap
		self.seed = seed

		self.path_memoizer = {}
			
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
		max_size = 64
		while current_tree_layer:
			if len(current_tree_layer) > max_size:
				lma_list = self._contract_tree(lma_list, dag, canonical_register)	
				if len(lma_list) > max_size:
					lma_list = list(np.random.choice(lma_list, size=max_size, replace=False))
				current_tree_layer = copy(lma_list)
			next_tree_layer = []
			invalidation_list = []
			for tree_node in current_tree_layer:
				if tree_node.front_layer:
					children, invalids = self.exec_on_subtree(tree_node, dag, canonical_register)
					for (i, c) in enumerate(children):
						next_tree_layer.append(c)
						if c.is_lma:
							lma_list.append(c)
					invalidation_list.extend(invalids)
				else:
					# Apply output list and exit.
					mapped_dag = dag._copy_circuit_metadata()
					for node in tree_node.output_list:
						mapped_dag.apply_operation_back(node.op, node.qargs, node.cargs)
					if min_dag is None or mapped_dag.size() < min_dag.size() or (mapped_dag.size() == min_dag.size() and mapped_dag.depth() < min_dag.depth()):
						min_dag = mapped_dag
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
					score = self._distf(tree_node.front_layer, post_layers, tree_node.current_layout)\
						+ np.sqrt(len(swap_list_cpy) + len(tree_node.output_list))*0.1
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
						#elif min_score == score:
						#	min_leaves.append(tree_node)
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
	
	def _get_post_layers(self, tree_node, dag, depth=5):
		post_layers = [[]]

		mnm = tree_node.marked_node_map.copy()
		d = 0
		curr_layer = tree_node.front_layer
		while d < depth and curr_layer:
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
			curr_layer = next_layer
			if post:
				post_layers.append(post)
			d += 1
		return post_layers	
				
	def _path_select(self, source, sink, max_length=None):
		if max_length is None:
			max_length = self.max_swaps

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
				if len(prev) + self.coupling_map.distance_matrix[w, sink] > max_length:
					continue
				elif len(prev) > 0 and prev[-1] == edge:
					continue
				prev_cpy = copy(prev)
				prev_cpy.append(edge)
				bfs_queue.append((w, prev_cpy))
		self.path_memoizer[memoizer_entry] = path_candidates
		return path_candidates
	
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
			dist += self.coupling_map.distance_matrix[current_layout[q0], current_layout[q1]]
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
				sub_sum += self.coupling_map.distance_matrix[current_layout[q0], current_layout[q1]]
			post_dist += sub_sum
		if num_ops > 0:
			dist += post_dist*0.5/num_ops
		return dist

