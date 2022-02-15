"""
    author: Suhas Vittal
    date:   20 October 2021
"""

class SumTreeNode:
    def __init__(self, obj_data, sum_data, parent, children, swap_segments=None, alap_used=0, asap_used=0):
        self.obj_data = obj_data
        self.sum_data = sum_data
        self.parent = parent
        self.children = children
        self.swap_segments = [] if swap_segments is None else swap_segments
        self.alap_used = alap_used
        self.asap_used = asap_used

class MinLeafPackage:
    def __init__(self, leaf_node, leaf_sum, layout, completed_nodes, alap_used, asap_used, kernel_type):
        self.leaf_node = leaf_node
        self.leaf_sum = leaf_sum
        self.layout = layout
        self.completed_nodes = completed_nodes
        self.alap_used = alap_used
        self.asap_used = asap_used
        self.type = kernel_type
