"""
	author: Suhas Vittal
	date:	20 October 2021
"""

class SumTreeNode:
	def __init__(self, obj_data, sum_data, parent, children):
		self.obj_data = obj_data
		self.sum_data = sum_data
		self.parent = parent
		self.children = children
