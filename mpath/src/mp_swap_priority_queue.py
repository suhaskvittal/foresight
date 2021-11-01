"""
	author: Suhas Vittal
	date:	20 October 2021
"""

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

