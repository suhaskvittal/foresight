"""
	author: Suhas Vittal
	date:	10 October 2021 @ 1:32 p.m. EST
"""

from qiskit.transpiler.basepasses import AnalysisPass

from timeit import default_timer as timer

class LayerViewPass(AnalysisPass):
	def __init__(self):
		super().__init__()

	def run(self, dag):
		start = timer()
		front_layer = dag.front_layer()
		visited = set()

		primary_layer_view = []
		secondary_layer_view = []
		# Traverse front layer to compute primary and secondary views. 
		# Primary view: 2-qubit ops.
		# Secondary view: other ops (added to output layer automatically).
		while front_layer:
			next_layer = []
			primary_bfs_layer = []
			secondary_bfs_layer = []

			for op in front_layer:
				if op.type != 'op' or op in visited:	
					continue
				if _is_bad_op(op):
					secondary_bfs_layer.append(op)
					visited.add(op)
				else:
					primary_bfs_layer.append(op)	
					visited.add(op)
			added = set()
			for bfs_layer in [primary_bfs_layer, secondary_bfs_layer]:
				for op in bfs_layer:
					for child_op in dag.descendants(op):
						# IF child is NOT(bad OR visited) AND all ancestors are visited THEN add to next layer 
						if child_op not in added and all(x in visited for x in dag.ancestors(child_op) if x.type == 'op'):
							next_layer.append(child_op)
							added.add(child_op)
			# Push bfs layer onto layer view
			primary_layer_view.append(primary_bfs_layer)
			secondary_layer_view.append(secondary_bfs_layer)
			# Update front layer
			front_layer = next_layer
		end = timer()
		self.property_set['primary_layer_view'] = primary_layer_view
		self.property_set['secondary_layer_view'] = secondary_layer_view
		self.property_set['bench_layer_view'] = end - start

def _is_bad_op(op):
	return len(op.qargs) != 2 or op.name == 'measure'
