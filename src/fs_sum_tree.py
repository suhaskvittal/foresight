"""
    author: Suhas Vittal
    date:   20 October 2021
"""

class SumTreeNode:
    def __init__(self, obj_data, sum_data, parent, children, swap_segments=None):
        self.obj_data = obj_data
        self.sum_data = sum_data
        self.parent = parent
        self.children = children
        self.swap_segments = [] if swap_segments is None else swap_segments

class MinLeafPackage:
    def __init__(self, leaf_node, leaf_sum, layout, completed_nodes):
        self.leaf_node = leaf_node
        self.leaf_sum = leaf_sum
        self.layout = layout
        self.completed_nodes = completed_nodes
