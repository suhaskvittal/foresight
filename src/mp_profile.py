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

from mp_benchmark_pass import BenchmarkPass
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

		print('\trunning profiler')
		profile_pm.run(circ)
		benchmark_results = benchmark_pass.benchmark_results
		data['improvement'].append((benchmark_results['SABRE CNOTs'] - benchmark_results['MPATH_IPS CNOTs']) / (benchmark_results['SABRE CNOTs']))
		data['layer density, mean'].append(benchmark_results['Layer Density, mean'])
		data['layer density, std'].append(benchmark_results['Layer Density, std.'])
		data['child distance, mean'].append(benchmark_results['Child Distance, mean'])
		data['child distance, std'].append(benchmark_results['Child Distance, std.'])
		# Measure improvement
		# Logging
		for x in data:
			print('\t%s: %.3f' % (x, data[x][-1]))
	df = pd.DataFrame(data=data, index=used_benchmarks)
	df.to_csv(out_file)
				
if __name__ == '__main__':
	out_file = argv[1]
	profile_optimal_ips_workload(out_file)
				
