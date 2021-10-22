"""
	author: Suhas Vittal
	date: 	21 October 2021
"""
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *

import numpy as np
import pandas as pd

from mp_benchmark import BenchmarkPass
from mp_util import G_IBM_TORONTO,\
					G_QISKIT_GATE_SET,\
					G_ZULEHNER
from mp_exec import _pad_circuit_to_fit

from copy import copy
from sys import argv
from collections import defaultdict

def profile_optimal_ips_workload(out_file):
	coupling_map = G_IBM_TORONTO
	data = defaultdict(list)	

	benchmark_pass = BenchmarkPass(coupling_map, compare=['sabre', 'ips'], runs=1)
	profile_pm = PassManager([
		Unroller(G_QISKIT_GATE_SET),
		SabreLayout(coupling_map, routing_pass=SabreSwap(coupling_map, heuristic='decay')),
		ApplyLayout(),
		benchmark_pass
	])

	used_benchmarks = []
	for qb_file in G_ZULEHNER:
		circ = QuantumCircuit.from_qasm_file('benchmarks/zulehner/%s' % qb_file)
		if circ.depth() > 2000 or circ.depth() < 100:
			continue
		used_benchmarks.append(qb_file)
		_pad_circuit_to_fit(circ, coupling_map)
		print('[%s]' % qb_file)
		data['size'].append(circ.size())
		data['depth'].append(circ.depth())

		density_mean, density_std = compute_per_layer_density_2q(circ)
		data['layer density, mean'].append(density_mean)
		data['layer density, std'].append(density_std)
		child_distance_mean, child_distance_std = compute_child_distance_2q(circ)
		data['child distance, mean'].append(child_distance_mean)
		data['child distance, std'].append(child_distance_std)
		# Measure improvement
		print('\trunning profiler')
		profile_pm.run(circ)
		benchmark_results = benchmark_pass.benchmark_results
		data['improvement'].append((benchmark_results['SABRE CNOTs'] - benchmark_results['MPATH_IPS CNOTs']) / (benchmark_results['SABRE CNOTs']))
		# Logging
		for x in data:
			print('\t%s: %.3f' % (x, data[x][-1]))
	df = pd.DataFrame(data=data, index=used_benchmarks)
	df.to_csv(out_file)
	
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
				
if __name__ == '__main__':
	out_file = argv[1]
	profile_optimal_ips_workload(out_file)
				
