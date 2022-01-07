"""
    author: Suhas Vittal
    date:   7 January 2022
"""

class ComputationKernel:
    def __init__(self, layout, parent_id, completed_nodes):
        self.layout = layout
        self.parent_id = parent_id
        self.completed_nodes = completed_nodes

class DeepSolveSolution:
    def __init__(self, output_layers, layout, layer_sum, completed_nodes):
        self.output_layers = output_layers
        self.layout = layout
        self.layer_sum = layer_sum
        self.completed_nodes = completed_nodes

class ShallowSolveSolution:
    def __init__(self, output_layers, layout, num_swaps, completed_nodes):
        self.output_layers = output_layers
        self.layout = layout
        self.num_swaps = num_swaps
        self.completed_nodes = completed_nodes
