"""
	author: Suhas Vittal
	date:	24 October 2021
"""

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import *

from timeit import default_timer as timer
from copy import copy, deepcopy

from mp_layerview import LayerViewPass
from mp_ips import MPATH_IPS
from mp_bsp import MPATH_BSP
from mp_hybrid import MPATH_HYBRID
from mp_hybrid_sabreswap import MPATH_HYBRID_SabreSwap
from mp_hybrid_ips import MPATH_HYBRID_IPS
from mp_exec import _bench_and_cmp, _pad_circuit_to_fit, draw
from mp_util import G_QISKIT_GATE_SET,\
					G_MPATH_IPS_SLACK,\
					G_MPATH_IPS_SOLN_CAP,\
					G_MPATH_BSP_TREE_WIDTH

import numpy as np

from collections import defaultdict

class BenchmarkPass(AnalysisPass):
	def __init__(self, coupling_map, hybrid_data_file, compare=['sabre', 'ips', 'bsp', 'hybrid'], runs=5):
		super().__init__()

		self.benchmark_list = compare

		sabre_router = SabreSwap(coupling_map, heuristic='decay')
		ips_router = MPATH_IPS(
				coupling_map,
				slack=G_MPATH_IPS_SLACK,
				solution_cap=G_MPATH_IPS_SOLN_CAP
			)
		bsp_router = MPATH_BSP(
				coupling_map,
				tree_width_limit=G_MPATH_BSP_TREE_WIDTH
			)
		# Declare hybrid routers.
		hybrid_sabre = MPATH_HYBRID_SabreSwap(coupling_map, heuristic='decay')
		hybrid_ips = MPATH_HYBRID_IPS(
				coupling_map,
				slack=G_MPATH_IPS_SLACK,
				solution_cap=G_MPATH_IPS_SOLN_CAP
			)
		hybrid_router = MPATH_HYBRID(hybrid_sabre, hybrid_ips, bsp_router, hybrid_data_file)

		self.sabre_pass = PassManager([
			sabre_router,
			Unroller(G_QISKIT_GATE_SET)
		])
		self.ips_pass = PassManager([
			ips_router,
			Unroller(G_QISKIT_GATE_SET)
		])
		self.bsp_pass = PassManager([
			bsp_router,
			Unroller(G_QISKIT_GATE_SET)
		])
		self.hybrid_pass = PassManager([
			hybrid_router,
			Unroller(G_QISKIT_GATE_SET)
		])
		self.look_pass = PassManager([
			LookaheadSwap(coupling_map, search_depth=4, search_width=4),
			Unroller(G_QISKIT_GATE_SET)
		])
		self.runs = runs

		self.benchmark_results = None
	
	def run(self, dag):
		original_circuit = dag_to_circuit(dag)
		original_circuit_cx = original_circuit.count_ops()['cx']

		self.benchmark_results = defaultdict(float)

		for r in range(self.runs):
			# Run dag on both passes. 
			# SABRE
			if 'sabre' in self.benchmark_list:
				start = timer()
				sabre_circ = self.sabre_pass.run(original_circuit)
				end = timer()
				sabre_cnots = sabre_circ.count_ops()['cx'] - original_circuit_cx
				if r == 0 or self.benchmark_results['SABRE CNOTs'] > sabre_cnots:
					self.benchmark_results['SABRE CNOTs'] = sabre_cnots
					self.benchmark_results['SABRE Depth'] = (sabre_circ.depth())
					self.benchmark_results['SABRE Time'] = (end - start)
			# MPATH IPS
			if 'ips' in self.benchmark_list:
				start = timer()
				ips_circ = self.ips_pass.run(original_circuit)
				end = timer()
				ips_cnots = ips_circ.count_ops()['cx'] - original_circuit_cx
				if r == 0 or self.benchmark_results['MPATH_IPS CNOTs'] > ips_cnots:
					self.benchmark_results['MPATH_IPS CNOTs'] = ips_cnots
					self.benchmark_results['MPATH_IPS Depth'] = (ips_circ.depth())
					self.benchmark_results['MPATH_IPS Time'] = (end - start)
			# MPATH BSP
			if 'bsp' in self.benchmark_list:
				start = timer()
				bsp_circ = self.bsp_pass.run(original_circuit)
				end = timer()
				bsp_cnots = bsp_circ.count_ops()['cx'] - original_circuit_cx
				if r == 0 or self.benchmark_results['MPATH_BSP CNOTs'] > bsp_cnots:
					self.benchmark_results['MPATH_BSP CNOTs'] = bsp_cnots
					self.benchmark_results['MPATH_BSP Depth'] = (bsp_circ.depth())
					self.benchmark_results['MPATH_BSP Time'] = (end - start)
			# MPATH HYBRID
			if 'hybrid' in self.benchmark_list:
				start = timer()
				hybrid_circ = self.hybrid_pass.run(original_circuit)
				end = timer()
				hybrid_cnots = hybrid_circ.count_ops()['cx'] - original_circuit_cx
				if r == 0 or self.benchmark_results['MPATH_HYBRID CNOTs'] > hybrid_cnots:
					self.benchmark_results['MPATH_HYBRID CNOTs'] = hybrid_cnots
					self.benchmark_results['MPATH_HYBRID Depth'] = (hybrid_circ.depth())
					self.benchmark_results['MPATH_HYBRID Time'] = (end - start)
			# LOOK
			if 'look' in self.benchmark_list:
				try:
					start = timer()
					look_circ = self.look_pass.run(original_circuit)
					end = timer()
					if self.benchmark_results['LOOK CNOTs'] >= 0:
						self.benchmark_results['LOOK CNOTs'] += (look_circ.count_ops()['cx'] - original_circuit_cx)
						self.benchmark_results['LOOK Depth'] += (look_circ.depth())
						self.benchmark_results['LOOK Time'] += (end - start)
				except TranspilerError:
					self.benchmark_results['LOOK CNOTs'] = -1.0
					self.benchmark_results['LOOK Depth'] = -1.0
					self.benchmark_results['LOOK Time'] = -1.0
		# Some circuit statistics as well.
		self.benchmark_results['Layer Density, mean'], self.benchmark_results['Layer Density, std.'] =\
			compute_per_layer_density_2q(original_circuit)
		self.benchmark_results['Child Distance, mean'], self.benchmark_results['Child Distance, std.'] =\
			compute_child_distance_2q(original_circuit)
	
def compute_per_layer_density_2q(circuit):
	dag = circuit_to_dag(circuit) 
	dag_layers = dag.layers()
	densities = []
	for layer in dag_layers:
		densities.append(len([node for node in layer['graph'].front_layer() if node.type == 'op' and len(node.qargs) == 2]))
	return np.mean(densities), np.std(densities)

def compute_child_distance_2q(circuit):
	dag = circuit_to_dag(circuit)
	dag_layers = dag.layers()
	child_mark_map = {}

	node_to_parent = {}
	for node in dag.front_layer():
		if node.type == 'op' and len(node.qargs) == 2:
			q0, q1 = node.qargs
			node_to_parent[q0] = (0, node)
			node_to_parent[q1] = (0, node)
	n = 0
	child_distances = []
	num_layers = 0
	for layer in dag_layers:
		if num_layers == 0:
			num_layers += 1
			continue
		for child in layer['graph'].front_layer():
			if child.type != 'op' or len(child.qargs) != 2:
				continue
			q0, q1 = child.qargs
			left_parent = None
			if q0 in node_to_parent:
				home_layer, parent = node_to_parent[q0]
				child_distances.append(num_layers - home_layer)
				left_parent = parent
			node_to_parent[q0] = (num_layers, child)
			if q1 in node_to_parent and left_parent != node_to_parent[q1][1]:
				home_layer, parent = node_to_parent[q1]
				child_distances.append(num_layers - home_layer)
			node_to_parent[q1] = (num_layers, child)
		num_layers += 1
	return np.mean(child_distances), np.std(child_distances)
