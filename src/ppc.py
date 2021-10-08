"""
	author: Suhas Vittal
	date:	4 October 2021 @ 11:08 a.m. EST
"""

import numpy as np

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
		for _ in range(runs):
			collection = []
			size = 0
			conflict_matrix_line = [0 for _ in range(self.num_vertices)]			
			score_sum = 0.0
			for (i, pq) in enumerate(self.pqueues):
				path, s = pq.peek()
				# Otherwise, check for conflicts.
				shift_ctr = self._shift_or_fail(conflict_matrix_line, self.conflict_matrix[i], path) 
				if shift_ctr < 0:  # Error has occurred, reduce score of path.
					size = -1  # Invalidate score.
					pq.change_score(path, s * 1.5, self.conflict_matrix, i)  # Penalize path.
					break
				# Perform shift and update conflict_matrix_line.
				for (j, (v0, v1)) in enumerate(path):
					conflict_matrix_line[v0] |= self.conflict_matrix[i][v0] << shift_ctr	
					conflict_matrix_line[v1] |= self.conflict_matrix[i][v1] << shift_ctr
					if j + shift_ctr < len(collection) and path[j] not in collection[j+shift_ctr]:
						collection[j+shift_ctr].append(path[j])
					else:
						collection.append([path[j]])
					size += 1
				# Finally, randomly increase score of path in pq.
				score_sum += s
				if np.random.random() < 0.3:
					pq.change_score(path, s * 1.5, self.conflict_matrix, i) 
			if size < 0 or _soln_hash_f(collection) in soln_hash:
				continue
			soln_hash.add(_soln_hash_f(collection))
			is_valid, score = verifier._verify_and_measure(collection, target_set, current_layout, post_primary_layer_view)
			if score == score_sum:
				print('MATCH')
			# If we get to this point, verify and update.
			if is_valid:
				if min_score < 0 or score < min_score:
					min_score = score
					min_collection_list = [collection]
				elif score == min_score:
					min_collection_list.append(collection)
		return min_collection_list
	
	def _shift_or_fail(self, curr_line, next_line, next_path):
		shift_ctr = 0
		for (i, (v0, v1)) in enumerate(next_path):
			# Bring down v0 and v1 line.
			curr_v0_line, curr_v1_line = curr_line[v0], curr_line[v1]
			next_v0_line, next_v1_line = next_line[v0] << shift_ctr, next_line[v1] << shift_ctr
			while curr_v0_line & next_v0_line != 0 and curr_v1_line & next_v1_line != 0:
				if shift_ctr == self.max_shift_ctr:
					return -1  # Failure
				shift_ctr += 1
				next_v0_line <<= 1
				next_v1_line <<= 1
		return shift_ctr
	
class PriorityQueue:
	def __init__(self):
		self.backing_array = [0]
		self.size = 0
	
	def buildheap(array, conflict_matrix, heap_index):
		pq = PriorityQueue()
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

def _is_pow2(x):
	return (x & (x-1)) == 0
	
def _path_hash_f(path):
	h = 0
	PRIME = 5586537595543
	for (p0, p1) in path:
		h += ((2**p0)*(3**p1)) % PRIME
	return h	

def _soln_hash_f(soln):
	h = 0
	PRIME = 5586537595543
	for (i, layer) in enumerate(soln):
		for (p0, p1) in layer:
			h += ((2**p0)*(3**p1)*(5**i)) % PRIME
	return h

