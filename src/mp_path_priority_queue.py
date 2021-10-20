"""
	author: Suhas Vittal
	date:	20 October 2021
"""

from mp_util import _path_hash_f

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

