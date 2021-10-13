"""
	author: Suhas Vittal
	date:	4 October 2021 @ 11:08 a.m. EST
"""

import numpy as np

from dstruct import PathPriorityQueue, PathJoinTree, _soln_hash_f, _path_to_swap_collection

DEFAULT_SIZE = 20

class PriorityPathCollection:
	def __init__(self, path_collection_list, num_vertices, num_sources):
		self.num_vertices = num_vertices
		self.num_sources = num_sources
		self.conflict_matrix = [[0 for _ in range(num_vertices)] for _ in range(num_sources)]
		self.path_collection_list = path_collection_list
		self.pqueues = [PathPriorityQueue.buildheap(path_collection_list[i], self.conflict_matrix, i) for i in range(num_sources)]
	
	def find_and_join(self, verifier, target_list, current_layout, post_primary_layer_view, runs=1000):
		min_collection_list = []	
		min_score = -1

		if len(target_list) == 0:
			return [], None
		if len(target_list) == 1:
			pq = self.pqueues[0]
			while pq.size > 0:
				path, s = pq.dequeue(self.conflict_matrix, 0)
				if min_score < 0 or s < min_score:
					min_score = s
					min_collection_list.append(_path_to_swap_collection(path))
				elif s == min_score:
					min_collection_list.append(_path_to_swap_collection(path))
				else:
					break  # This is a heap, we will not get any other elements.
			return min_collection_list, None
		
		soln_hash = set()
		path_join_tree = PathJoinTree([pq.peek()[0] for pq in self.pqueues], self.conflict_matrix, verifier, target_list, current_layout, post_primary_layer_view)
		for r in range(runs):
			root = path_join_tree.root
			collection = root.data
			is_valid, score = root.valid, root.score 
			# Modify heaps and path join tree randomly.
			# If invalid, then find leaves contributing to invalid nodes.
			if not is_valid: # Search for invalid nodes.
				self._dfs_search_invalid_and_update(path_join_tree, verifier, target_list, current_layout, post_primary_layer_view)
				continue
			# Otherwise, just do random adjustments.
			self._random_select_and_update(path_join_tree, verifier, target_list, current_layout, post_primary_layer_view)	
			if _soln_hash_f(collection) in soln_hash:
				continue
			soln_hash.add(_soln_hash_f(collection))
			if is_valid and min_score > 0 and score > min_score:
				# The cost of searching for a golden goose is too high. Just exit.
				break
			if min_score < 0 or score < min_score:
				min_score = score
				min_collection_list.append(collection)
			elif score == min_score:
				min_collection_list.append(collection)
		if len(min_collection_list) == 0:  
			# Reset the PPC to the first tree.
			self.reset()
			# Modify all the leaves.
			modified_heaps = list(range(self.num_sources))
			new_paths = [pq.peek()[0] for pq in self.pqueues]
			path_join_tree.modify_leaves(modified_heaps, new_paths, verifier, target_list, current_layout, post_primary_layer_view)
			# Use DFS to suggest splits in the target list.
			dfs_stack = [path_join_tree.root]
			suggestions = []
			while dfs_stack:
				node = dfs_stack.pop()
				if not node.valid:
					dfs_stack.append(node.left_child)
					dfs_stack.append(node.right_child)
				else:
					suggestions.append(node.target_index_list)
			return None, suggestions

		return min_collection_list, None
	
	def reset(self):
		self.conflict_matrix = [[0 for _ in range(self.num_vertices)] for _ in range(self.num_sources)]
		self.pqueues = [PathPriorityQueue.buildheap(self.path_collection_list[i], self.conflict_matrix, i) for i in range(self.num_sources)]
	
	def _random_select_and_update(self, path_join_tree, verifier, target_list, current_layout, post_primary_layer_view):
		modified_heaps = []
		new_paths = []
		for (i, pq) in enumerate(self.pqueues):
			# Randomly adjust the scores of used paths in the segment tree.
			if np.random.random() < 0.5:
				path, s = pq.peek()
				pq.change_score(path, s * 10, self.conflict_matrix, i) 
				modified_heaps.append(i)
				new_paths.append(pq.peek()[0])
		if len(modified_heaps) > 0:
			path_join_tree.modify_leaves(modified_heaps, new_paths, verifier, target_list, current_layout, post_primary_layer_view)

	def _dfs_search_invalid_and_update(self, path_join_tree, verifier, target_list, current_layout, post_primary_layer_view):
		dfs_stack = [path_join_tree.root]

		modified_heaps = []
		new_paths = []
#		print('======')
		while dfs_stack:
			node = dfs_stack.pop()
			left, right = node.left_child, node.right_child
			# Only go down invalid children.
			# If neither are invalid, then update a random entry in the target_sub_list.
#			print(node.target_index_list, node.valid, left.target_index_list, left.valid, right.target_index_list, right.valid)
			if left.valid == 1 and right.valid == 1:
				update_index = node.target_index_list[np.random.randint(0, high=len(node.target_index_list))]
				pq = self.pqueues[update_index]
				path, s = pq.peek()
				pq.change_score(path, s * 10, self.conflict_matrix, update_index)
				modified_heaps.append(update_index)
				new_paths.append(pq.peek()[0])
			else:
				if left.valid == 0:
					dfs_stack.append(left)
				if right.valid == 0:
					dfs_stack.append(right)
		# Modify path join tree.
		if len(modified_heaps) > 0:
			path_join_tree.modify_leaves(modified_heaps, new_paths, verifier, target_list, current_layout, post_primary_layer_view)

