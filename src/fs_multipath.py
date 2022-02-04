"""
    author: Suhas Vittal
    date:   7 January 2022
"""

class ComputationKernel:
    def __init__(self, layout, parent_id, completed_nodes, kernel_type='alap'):
        self.layout = layout
        self.parent_id = parent_id
        self.completed_nodes = completed_nodes
        self.type = kernel_type

class DeepSolveSolution:
    def __init__(self, output_layers, layout, layer_sum, completed_nodes, swap_segments=None):
        self.output_layers = output_layers
        self.layout = layout
        self.layer_sum = layer_sum
        self.completed_nodes = completed_nodes
        self.swap_segments = [] if swap_segments is None else swap_segments

class ShallowSolveSolution:
    def __init__(self, output_layers, layout, num_swaps, completed_nodes):
        self.output_layers = output_layers
        self.layout = layout
        self.num_swaps = num_swaps
        self.completed_nodes = completed_nodes
