"""
	author: Suhas Vittal
	date:	20 October 2021
"""

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

