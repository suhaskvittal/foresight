"""
	author: Suhas Vittal
	date:	4 October 2021 @ 11:08 a.m. EST
"""

import numpy as np

from dstruct import PriorityQueue, SegmentTree, _soln_hash_f

DEFAULT_SIZE = 20

class PriorityPathCollection:
	def __init__(self, path_collection_list, num_vertices, num_sources, max_shift_ctr=10):
		self.num_vertices = num_vertices
		self.num_sources = num_sources
		self.max_shift_ctr = max_shift_ctr
		self.conflict_matrix = [[0 for _ in range(num_vertices)] for _ in range(num_sources)]
		self.pqueues = [PriorityQueue.buildheap(path_collection_list[i], self.conflict_matrix, i) for i in range(num_sources)]
	
	def find_and_join(self, verifier, target_set, current_layout, post_primary_layer_view, runs=500):
		min_collection_list = []	
		min_score = -1
		
		soln_hash = set()
		segment_tree = SegmentTree([pq.peek()[0] for pq in self.pqueues], self.conflict_matrix)
		for run in range(runs):
			collection = segment_tree.get_root()
			modified_heaps = []
			new_paths = []
			for (i, pq) in enumerate(self.pqueues):
				# Randomly adjust the scores of used paths in the segment tree.
				if np.random.random() < 0.3:
					path, s = pq.peek()
					pq.change_score(path, s * 1.5, self.conflict_matrix, i) 
					modified_heaps.append(i)
					new_paths.append(pq.peek()[0])
			if len(modified_heaps) > 0:
				segment_tree.modify_leaves(modified_heaps, new_paths)
			if _soln_hash_f(collection) in soln_hash:
				continue
			soln_hash.add(_soln_hash_f(collection))
			is_valid, score = verifier._verify_and_measure(collection, target_set, current_layout, post_primary_layer_view)
			if min_score >= 0 and score > min_score:
				# The cost of searching for a golden goose is too high. Just exit.
				break
			# If we get to this point, verify and update.
			if is_valid:
				if min_score < 0 or score < min_score:
					min_score = score
					min_collection_list = [collection]
				elif score == min_score:
					min_collection_list.append(collection)
		return min_collection_list
	

