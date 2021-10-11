"""
	author: Suhas Vittal
	date:	7 October 2021 @ 1:33 p.m. EST
"""

import numpy as np

from copy import copy, deepcopy

class SumTreeNode:
	def __init__(self, obj_data, sum_data, parent, children):
		self.obj_data = obj_data
		self.sum_data = sum_data
		self.parent = parent
		self.children = children

class PathJoinTreeNode:
	def __init__(self, swap_collection, conflict_matrix_line, parent, left_child, right_child, target_index_list):
		self.data = swap_collection		# path(left) JOIN path(right)
		self.conflict_matrix_line = conflict_matrix_line
		self.parent = parent
		self.left_child = left_child
		self.right_child = right_child
		# Data signals.
		self.dirty = 0  	# 1 = dirty
		self.conflict = 0	# 1 = has conflict
		self.valid = True
		# Other data held by a node.
		self.target_index_list = target_index_list
		self.score = 0

class PathJoinTree:
	def __init__(self, leaves, conflict_matrix, verifier, target_list, current_layout, post_primary_layer_view):	
		self.leaves = [PathJoinTreeNode(_path_to_swap_collection(path), conflict_matrix[i], None, None, None, [i]) for (i, path) in enumerate(leaves)]
		zero_line = [0]*len(conflict_matrix[0])
		while not _is_pow2(len(self.leaves)):
			self.leaves.append(PathJoinTreeNode([], zero_line, None, None, None, []))
		# Build segmented tree from leaves using BFS-like algorithm.
		queue = [leaf for leaf in self.leaves]
		while len(queue) > 1:  # If len(queue) == 1, we only have the root inside.
			left, right = queue.pop(0), queue.pop(0)
			parent = PathJoinTreeNode(None, None, None, left, right, None)
			self._update_node(parent, verifier, target_list, current_layout, post_primary_layer_view)
			left.parent = parent
			right.parent = parent
			queue.append(parent)  # Build parent by joining children.
		self.root = queue[0]  # candidate solution
	
	def get_root(self):
		return self.root.data

	def modify_leaves(self, modified_indices, new_paths, verifier, target_list, current_layout, post_primary_layer_view):
		queue = []
		for (i, k) in enumerate(modified_indices):
			self.leaves[k].data = _path_to_swap_collection(new_paths[i])
			if self.leaves[k].parent is None:  # this is the root, just break.
				continue
			self.leaves[k].parent.dirty = 1
			queue.append(self.leaves[k].parent)
		while queue:
			node = queue.pop(0)
			if node.dirty == 0:  # already updated.
				continue
			# Else, perform update.
			self._update_node(node, verifier, target_list, current_layout, post_primary_layer_view)
			node.dirty = 0
			# Place parent of node in the queue, mark it as dirty.
			if node.parent is not None:
				node.parent.dirty = 1
				queue.append(node.parent)
	
	def _update_node(self, node, verifier, target_list, current_layout, post_primary_layer_view):
		left, right = node.left_child, node.right_child
		j_collection = deepcopy(left.data)
		j_conflict_matrix_line = [0]*len(left.conflict_matrix_line)
		j_target_index_list = copy(left.target_index_list)

		conflict_bit = 0
		
		j_target_index_list.extend(right.target_index_list)

		# Don't do any work if the join is not going to be valid anyways.
		if not (left.valid and right.valid):
			node.data = j_collection
			node.conflict_matrix_line = j_conflict_matrix_line 
			node.conflict = 1
			node.valid = False
			node.target_index_list = j_target_index_list
			return 

		for (j, coll_layer) in enumerate(right.data):
			for (v0, v1) in coll_layer:
				left_line_v0, left_line_v1 = left.conflict_matrix_line[v0], left.conflict_matrix_line[v1]
				right_line_v0, right_line_v1 = right.conflict_matrix_line[v0], right.conflict_matrix_line[v1]
				j_conflict_matrix_line[v0] = left_line_v0 | right_line_v0		
				j_conflict_matrix_line[v1] = left_line_v1 | right_line_v1

				# Use conflict lines to check for conflict.
				# If there is a conflict, let the verifier reject the candidate.
				has_v0 = left_line_v0 & right_line_v0 & (1 << j)
				has_v1 = left_line_v1 & right_line_v1 & (1 << j)
				if has_v0 | has_v1 == 0:  # No conflict.
					if j >= len(j_collection):
						j_collection.append([(v0, v1)])
					elif (v0, v1) not in j_collection[j]:
						j_collection[j].append((v0, v1))
				else:
					conflict_bit = 1	
		node.data = j_collection
		node.conflict_matrix_line = j_conflict_matrix_line
		node.conflict = conflict_bit
		node.valid, node.score = verifier._verify_and_measure(j_collection, [target_list[i] for i in j_target_index_list], current_layout, post_primary_layer_view) 
		node.target_index_list = j_target_index_list

class PathPriorityQueue:
	def __init__(self):
		self.backing_array = [0]
		self.size = 0
	
	def buildheap(array, conflict_matrix, heap_index):
		pq = PathPriorityQueue()
		pq.backing_array.extend(array)
		pq.size = len(array)

		i = len(pq.backing_array) >> 1
		while i >= 1:
			pq._downheap(i, conflict_matrix, heap_index)
			i -= 1
		return pq

	def enqueue(self, path, score, conflict_matrix, heap_index):
		self.backing_array.append((path, score))
		# Update size.
		self.size += 1
		# Make sure heap property is maintained.
		self._upheap(self.size, conflict_matrix, heap_index)
	
	def dequeue(self, conflict_matrix, heap_index):
		root = self.backing_array[1]
		# Replace root with last entry.
		self.backing_array[1] = self.backing_array.pop()
		self.size -= 1
		# Make sure heap property is maintained.
		self._downheap(1, conflict_matrix, heap_index)
		return root
	
	def peek(self):
		return self.backing_array[1]
	
	def change_score(self, target_path, new_score, conflict_matrix, heap_index):	
		target_hash = _path_hash_f(target_path)
		for i in range(1, self.size+1):
			path, score = self.backing_array[i]
			if _path_hash_f(path) == target_hash:  # close enough.
				self.backing_array[i] = path, new_score  
				if new_score < score:  # Perform upheap to maintain heap property
					self._upheap(i, conflict_matrix, heap_index)
				elif new_score > score:  # Perform downheap
					self._downheap(i, conflict_matrix, heap_index)
				return score
		return -1
	
	def _upheap(self, from_index, conflict_matrix, heap_index):
		path, score = self.backing_array[from_index]
		curr_index = from_index
		while curr_index > 1:
			parent_path, parent_score = self.backing_array[curr_index >> 1]
			if parent_score < score:
				break
			self.backing_array[curr_index >> 1], self.backing_array[curr_index] =\
				self.backing_array[curr_index], self.backing_array[curr_index >> 1]
			# Update index.
			curr_index = curr_index >> 1
		# Update conflict matrix
		_update_conflict_matrix(self.backing_array[1], conflict_matrix, heap_index)

	def _downheap(self, from_index, conflict_matrix, heap_index): 
		path, score = self.backing_array[from_index]
		curr_index = from_index
		while curr_index < self.size + 1:
			min_child, min_score = None, -1
			child_index = 0
			if (curr_index << 1) <= self.size:
				min_child, min_score = self.backing_array[curr_index << 1]
			if (curr_index << 1) + 1 <= self.size:
				tmp_child, tmp_score = self.backing_array[(curr_index << 1) + 1]
				if min_score < 0 or tmp_score < min_score:
					min_child, min_score = tmp_child, tmp_score
					child_index = 1
			if min_score >= 0 and min_score < score:  # swap with the child
				self.backing_array[curr_index], self.backing_array[(curr_index << 1) + child_index] =\
					self.backing_array[(curr_index << 1) + child_index], self.backing_array[curr_index]
				curr_index = (curr_index << 1) + child_index
			else:
				break
		# Update conflict matrix
		_update_conflict_matrix(self.backing_array[1], conflict_matrix, heap_index)

def _update_conflict_matrix(entry, conflict_matrix, heap_index):
	path, _ = entry

	i = 0
	# Clear conflict matrix entry.
	for i in range(len(conflict_matrix[heap_index])):
		conflict_matrix[heap_index][i] = 0
	# Now update it.
	for (i, (v0, v1)) in enumerate(path):
		conflict_matrix[heap_index][v0] |= 1 << i
		conflict_matrix[heap_index][v1] |= 1 << i

def _soln_hash_f(soln):
	h = 0
	PRIME = 5586537595543
	for (i, layer) in enumerate(soln):
		for (p0, p1) in layer:
			h += ((2**p0)*(3**p1)*(5**i)) % PRIME
	return h

def _path_hash_f(path):
	h = 0
	PRIME = 5586537595543
	for (p0, p1) in path:
		h += ((2**p0)*(3**p1)) % PRIME
	return h	

def _path_to_swap_collection(path):
	collection = []
	for (v1, v2) in path:
		collection.append([(v1, v2)])
	return collection

def _is_pow2(x):
	return (x & (x-1)) == 0
