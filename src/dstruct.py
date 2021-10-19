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
	def __init__(self, swap_collection, parent, left_child, right_child, target_index_list):
		self.data = swap_collection		# path(left) JOIN path(right)
		self.parent = parent
		self.left_child = left_child
		self.right_child = right_child
		# Data signals.
		self.dirty = 0  	# 1 = dirty
		self.valid = True
		# Other data held by a node.
		self.target_index_list = target_index_list
		self.score = 0

class PathJoinTree:
	def __init__(self, leaves, verifier, target_list, current_layout, post_primary_layer_view):	
		self.leaves = [PathJoinTreeNode(_path_to_swap_collection(path), None, None, None, [i]) for (i, path) in enumerate(leaves)]
		while not _is_pow2(len(self.leaves)):
			self.leaves.append(PathJoinTreeNode([], None, None, None, []))
		# Build segmented tree from leaves using BFS-like algorithm.
		queue = [leaf for leaf in self.leaves]
		while len(queue) > 1:  # If len(queue) == 1, we only have the root inside.
			left, right = queue.pop(0), queue.pop(0)
			parent = PathJoinTreeNode(None, None, left, right, None)
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
			if self.leaves[k].parent is None:  # this is the root
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
		j_collection = []
		j_target_index_list = copy(left.target_index_list)
		
		j_target_index_list.extend(right.target_index_list)

		# Don't do any work if the join is not going to be valid anyways.
		if not (left.valid and right.valid):
			node.data = j_collection
			node.valid = False
			node.target_index_list = j_target_index_list
			return 

		# Join by inserting SWAPs from right.data whenever we can. 
		# Do not preemtively continue to other layers if we are not done 
		right_data_ptr = 0
		left_data_ptr = 0
		while right_data_ptr < len(right.data):
			right_layer = right.data[right_data_ptr]
			while right_layer:
				if left_data_ptr >= len(left.data):
					j_collection.append(copy(right_layer))
					break
				left_layer = left.data[left_data_ptr]

				joined_layer = []
				visited = set()
				for (v0, v1) in left_layer:
					joined_layer.append((v0, v1))
					visited.add(v0)
					visited.add(v1)
				remaining_swaps = []	
				for (v0, v1) in right_layer:
					if v0 in visited or v1 in visited:
						remaining_swaps.append((v0, v1))
					else:
						joined_layer.append((v0, v1))
					visited.add(v0)
					visited.add(v1)
				j_collection.append(joined_layer)
				left_data_ptr += 1
				right_layer = remaining_swaps
			right_data_ptr += 1

		node.data = j_collection
		if node.parent is None:
			node.valid, node.score = verifier._verify_and_measure(
				j_collection, 
				[target_list[i] for i in j_target_index_list], 
				current_layout, 
				post_primary_layer_view, 
				verify_only=False
			) 
		else:
			node.valid, _ = verifier._verify_and_measure(
				j_collection, 
				[target_list[i] for i in j_target_index_list], 
				current_layout, 
				post_primary_layer_view, 
				verify_only=True
			) 
		node.target_index_list = j_target_index_list

class PathPriorityQueue:
	def __init__(self):
		self.backing_array = [0]
		self.size = 0
	
	def buildheap(array):
		pq = PathPriorityQueue()
		pq.backing_array.extend(array)
		pq.size = len(array)

		i = len(pq.backing_array) >> 1
		while i >= 1:
			pq._downheap(i)
			i -= 1
		return pq

	def enqueue(self, path, score):
		self.backing_array.append((path, score))
		orig_path, orig_score = self.backing_array[1]
		# Update size.
		self.size += 1
		# Make sure heap property is maintained.
		self._upheap(self.size)
	
	def dequeue(self):
		root = self.backing_array[1]
		self.size -= 1
		# Replace root with last entry.
		if self.size > 0:
			self.backing_array[1] = self.backing_array.pop()
			# Make sure heap property is maintained.
			self._downheap(1)
		return root
	
	def peek(self):
		return self.backing_array[1]
	
	def change_score(self, target_path, new_score):	
		target_hash = _path_hash_f(target_path)
		for i in range(1, self.size+1):
			path, score = self.backing_array[i]
			if _path_hash_f(path) == target_hash:  # close enough.
				self.backing_array[i] = path, new_score  
				if new_score < score:  # Perform upheap to maintain heap property
					self._upheap(i)
				elif new_score > score:  # Perform downheap
					self._downheap(i)
				return score
		return -1
	
	def _upheap(self, from_index):
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

	def _downheap(self, from_index): 
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
