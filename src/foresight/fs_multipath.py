"""
    author: Suhas Vittal
    date:   6 April 2022
"""

class SolutionKernel:
    def __init__(
        self, 
        front_layer,
        next_layer,
        pred_table,
        completed_nodes,
        schedule,
        layout,
        swap_count,
        expected_prob_success, 
        parent,
        swap_segments=None
    ):
        self.front_layer = front_layer
        self.next_layer = next_layer
        self.pred_table = pred_table
        self.completed_nodes = completed_nodes

        self.schedule = schedule
        self.layout = layout
        self.swap_count = swap_count
        self.expected_prob_success = expected_prob_success
        self.parent = parent
            
        self.swap_segments = [] if swap_segments is None else swap_segments 

