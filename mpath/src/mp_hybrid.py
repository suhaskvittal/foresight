"""

		node_to_parent = {}
		for node in primary_layer_view[0]:
			q0, q1 = node.qargs
			node_to_parent[q0] = (0, node)
			node_to_parent[q1] = (0, node)
		verified_parents = set()

	author: Suhas Vittal
	date:	26 October 2021
"""

from qiskit.circuit import Qubit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from mp_layerview import LayerViewPass
from mp_stat import load_classifier, load_regressor

import numpy as np

from copy import copy, deepcopy
from collections import deque, defaultdict

import warnings

class MPATH_HYBRID(TransformationPass):
	def __init__(self, sabre_router, ips_router, bsp_router, profile_file, window_size=50):
		super().__init__()
		self.sabre_router = sabre_router
		self.ips_router = ips_router
		self.bsp_router = bsp_router
		self.classifier = load_classifier(profile_file)
		self.window_size = window_size
		
		self.fake_run = False
		self.router_usage = defaultdict(int)
	
	def run(self, dag):
		mapped_dag = dag._copy_circuit_metadata()
		canonical_register = dag.qregs["q"]
		current_layout = Layout.generate_trivial_layout(canonical_register)
		
		warnings.filterwarnings("ignore")

		primary_layer_view, secondary_layer_view = self.property_set['primary_layer_view'], self.property_set['secondary_layer_view']
		if primary_layer_view is None or secondary_layer_view is None:
			layer_view_pass = LayerViewPass();
			layer_view_pass.run(dag)
			primary_layer_view, secondary_layer_view = layer_view_pass.property_set['primary_layer_view'], layer_view_pass.property_set['secondary_layer_view']
		primary_layer_view = deque(primary_layer_view)
		secondary_layer_view = deque(secondary_layer_view)

		self.router_usage = defaultdict(int)
		while primary_layer_view:
			current_layout = self._find_and_route_on_subgraph(
				mapped_dag,
				primary_layer_view,
				secondary_layer_view,
				current_layout,
				canonical_register
			)
		self.property_set['final_layout'] = current_layout
		return mapped_dag
	
	def _find_and_route_on_subgraph(self, base_dag, primary_layer_view, secondary_layer_view, current_layout, canonical_register):
		sub_dag = base_dag._copy_circuit_metadata()
		# Remove corresponding layers from primary and secondary layers.
		layer_densities = []
		child_distances = []

		num_2q_layers = 0
		i = 0

		prev_nodes = []
		best_router = None
		last_window_index = 0
		while True:
			if ((i+1) % self.window_size) == 0 or i == len(primary_layer_view):
				# Process window and predict routing algorithm.	
				if layer_densities:
					mean_density, std_density = np.mean(layer_densities), np.std(layer_densities)
				else:
					mean_density, std_density = 0, 0
				if child_distances:
					mean_child_dist, std_child_dist = np.mean(child_distances), np.std(child_distances)
				else:
					mean_child_dist, std_child_dist = 0, 0
				router_pred = self.classifier.predict([[
					mean_density,
					std_density,
					mean_child_dist,
					std_child_dist
				]])[0]
				if best_router is None or router_pred == best_router:
					best_router = router_pred
					# Add prev_nodes to the sub dag.
					for node in prev_nodes:
						mapped_node = self._remap_gate_for_layout(node, current_layout, canonical_register)
						sub_dag.apply_operation_back(op=mapped_node.op, qargs=mapped_node.qargs, cargs=mapped_node.cargs)
					last_window_index = i
				else:
					break
				# Reset all variables.
				layer_densities = []
				child_distances = []
				prev_nodes = []
			if i == len(primary_layer_view):
				break
			# Collect data
			layer_densities.append(len(primary_layer_view[i]))
			if i > 0 and ((i+1) % self.window_size) != 0:
				for node in primary_layer_view[i]:
					for q in node.qargs:
						if q in node_to_parent:
							home_layer, parent = node_to_parent[q]
							if parent not in verified_parents:
								child_distances.append(num_2q_layers - home_layer)	
						node_to_parent[q] = (num_2q_layers, node)
					prev_nodes.append(node)
			else:
				# Reset for new window.
				node_to_parent = {}
				for node in primary_layer_view[i]:
					q0, q1 = node.qargs
					node_to_parent[q0] = (num_2q_layers, node)
					node_to_parent[q1] = (num_2q_layers, node)
					prev_nodes.append(node)
				verified_parents = set()
			prev_nodes.extend(secondary_layer_view[i])
			num_2q_layers += 1 if len(primary_layer_view) > 0 else 0
			i += 1
		# Determine router.
		if best_router == 0:
			best_router = self.sabre_router
			self.router_usage['sabre'] += last_window_index
		elif best_router == 1:
			best_router = self.ips_router
			self.router_usage['ips'] += last_window_index
		# Pop the used layers from both layer views.
		for _ in range(last_window_index):
			primary_layer_view.popleft()
			secondary_layer_view.popleft()
		# Map the sub dag.
		best_router.future_layers = primary_layer_view
		mapped_sub_dag = best_router.run(sub_dag)
		base_dag.compose(mapped_sub_dag, inplace=True)
		return best_router.property_set['final_layout']
	
	def _remap_gate_for_layout(self, op, layout, canonical_register):
		new_op = copy(op)
		new_op.qargs = [canonical_register[layout[x]] for x in op.qargs]
		return new_op
