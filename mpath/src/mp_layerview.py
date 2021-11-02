"""
    author: Suhas Vittal
    date:   10 October 2021 @ 1:32 p.m. EST
"""

from qiskit.circuit.quantumregister import Qubit
from qiskit.transpiler.basepasses import AnalysisPass

from collections import defaultdict

from timeit import default_timer as timer

class LayerViewPass(AnalysisPass):
    def __init__(self):
        super().__init__()

    def run(self, dag):
        start = timer()

        primary_layer_view = []
        secondary_layer_view = []
        # Primary view: 2-qubit ops.
        # Secondary view: other ops (added to output layer automatically).
        front_layer = dag.front_layer()
        pred = defaultdict(int)
        # Initialize pred.
        for (_, input_node) in dag.input_map.items():
            for (_, child_op, edge_data) in dag.edges(input_node):
                if child_op.type == 'op' and isinstance(edge_data, Qubit):
                    pred[child_op] += 1

        while front_layer:
            next_front_layer = []
            primary_bfs_layer, secondary_bfs_layer = [], []
            exec_list = []
            for op in front_layer:
                if op.type != 'op':
                    continue
                elif _is_bad_op(op):
                    secondary_bfs_layer.append(op)
                else:
                    primary_bfs_layer.append(op)
                exec_list.append(op)
            for op in exec_list:
                for (_, child_op, edge_data) in dag.edges(op):
                    if child_op.type != 'op' or not isinstance(edge_data, Qubit):
                        continue
                    pred[child_op] += 1
                    if pred[child_op] == len(child_op.qargs):
                        next_front_layer.append(child_op)
            front_layer = next_front_layer
            primary_layer_view.append(primary_bfs_layer)
            secondary_layer_view.append(secondary_bfs_layer)
        end = timer()
        self.property_set['primary_layer_view'] = primary_layer_view
        self.property_set['secondary_layer_view'] = secondary_layer_view
        self.property_set['bench_layer_view'] = end - start

def _is_bad_op(op):
    return len(op.qargs) != 2 or op.name == 'measure'
