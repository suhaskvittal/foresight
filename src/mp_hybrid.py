"""
	author: Suhas Vittal
	date:	26 October 2021
"""

from qiskit.circuit import Qubit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from mp_layerview import LayerViewPass
from mp_stat import load_classifier

import numpy as np

from copy import copy, deepcopy
from collections import deque

import warnings

class MPATH_HYBRID(TransformationPass):
	def __init__(self, sabre_router, ips_router, bsp_router, profile_file, window_size=100):
		super().__init__()
		self.sabre_router = sabre_router
		self.ips_router = ips_router
		self.bsp_router = bsp_router
		self.classifier = load_classifier(profile_file)
		self.window_size = window_size
		
		self.fake_run = False
	
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
		operations = []
		for _ in range(min(len(primary_layer_view), self.window_size)):
			operations.extend(primary_layer_view.popleft())
			operations.extend(secondary_layer_view.popleft())
		for node in operations:
			mapped_node = self._remap_gate_for_layout(node, current_layout, canonical_register)
			sub_dag.apply_operation_back(op=mapped_node.op, qargs=mapped_node.qargs, cargs = mapped_node.cargs)
		min_dag = None
		min_layout = None
		for router in [self.sabre_router, self.ips_router]:
			router.future_layers = primary_layer_view
			routed_dag = router.run(sub_dag)
			if min_dag is None or routed_dag.size() < min_dag.size() or (routed_dag.size() == min_dag.size() and routed_dag.depth() < min_dag.depth()):
				min_dag = routed_dag
				min_layout = router.property_set['final_layout']
		# Push routed dag onto the existing dag.
		base_dag.compose(min_dag, inplace=True)
		return min_layout
	
	def _remap_gate_for_layout(self, op, layout, canonical_register):
		new_op = copy(op)
		new_op.qargs = [canonical_register[layout[x]] for x in op.qargs]
		return new_op
