"""
	author: Suhas Vittal
	date:	10 October 2021 @ 1:32 p.m. EST
"""

from qiskit.circuit.quantumregister import Qubit
from qiskit.transpiler.basepasses import AnalysisPass

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
		for layer in dag.layers():
			primary_bfs_layer, secondary_bfs_layer = [], []
			for op in layer['graph'].front_layer():
				if op.type != 'op':
					continue
				elif _is_bad_op(op):
					secondary_bfs_layer.append(op)
				else:
					primary_bfs_layer.append(op)
			primary_layer_view.append(primary_bfs_layer)
			secondary_layer_view.append(secondary_bfs_layer)
		end = timer()
		self.property_set['primary_layer_view'] = primary_layer_view
		self.property_set['secondary_layer_view'] = secondary_layer_view
		self.property_set['bench_layer_view'] = end - start

def _is_bad_op(op):
	return len(op.qargs) != 2 or op.name == 'measure'
