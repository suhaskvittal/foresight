"""
	author: Suhas Vittal
	date:	5 October 2021 @ 2:35 p.m. EST
"""

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import Aer

from timeit import default_timer as timer
from copy import copy, deepcopy

from mpathswap import MultipathSwap
from layerview import LayerViewPass
from qcirc import _bench_and_cmp, _build_pass_manager, _pad_circuit_to_fit

import pandas as pd
import numpy as np

from sys import argv

from os import listdir
from os.path import isfile, join

G_QISKIT_OPT_NONE = 0
G_QISKIT_OPT_LIGHT = 1
G_QISKIT_OPT_HEAVY = 2
G_QISKIT_OPT_HEAVIER = 3

G_QISKIT_OPT_LVL = G_QISKIT_OPT_HEAVY  
G_QISKIT_GATE_SET = ['u1', 'u2', 'u3', 'cx']

ibm_toronto = np.array([[0, 1], [1, 0], [1, 2], [1, 4], [2, 1], [2, 3], [3, 2], [3, 5], [4, 1], [4, 7], [5, 3], [5, 8], [6, 7], [7, 4], [7, 6], [7, 10], [8, 5], [8, 9], [8, 11], [9, 8], [10, 7], [10, 12], [11, 8], [11, 14], [12, 10], [12, 13], [12, 15], [13, 12], [13, 14], [14, 11], [14, 13], [14, 16], [15, 12], [15, 18], [16, 14], [16, 19], [17, 18], [18, 15], [18, 17], [18, 21], [19, 16], [19, 20], [19, 22], [20, 19], [21, 18], [21, 23], [22, 19], [22, 25], [23, 21], [23, 24], [24, 23], [24, 25], [25, 22], [25, 24], [25, 26], [26, 25]])

qasmbench_medium = [
	'adder_n10',		# single adder
	'qft_n15',
	'dnn_n8',			# quantum deep neural net
	'cc_n12',			# counterfeit coin
	'multiplier_n15', 	# binary multiplier
	'qf21_n15', 		# quantum phase estimation, factor 21			
	'sat_n11',		
	'seca_n11',			# shor's error correction
	'bv_n14',			# bernstein-vazirani algorithm 
	'ising_n10',		# ising gate sim
	'qaoa_n6',			
	'qpe_n9',			# quantum phase estimation
	'simon_n6'			# simon's algorithm	
]

qasmbench_large = [
	'bigadder_n18',		# ripple carry adder	
	'qft_n20',
	'ising_n26',		# ising gate sim
	'bv_n19',			# bernstein-vazirani algorithm	
	'dnn_n16',			# quantum deep neural net
	'multiplier_n25',	# binary multiplier
	'wstate_n27',		
	'ghz_state_n23',
	'cat_state_n22',	
	'square_root_n18',  # square root	
	'cc_n18'			# counterfeit coin
]

zulehner = [f for f in listdir('benchmarks/zulehner') if isfile(join('benchmarks/zulehner', f))]

class BenchmarkPass(AnalysisPass):
	def __init__(self, mpath_mapping_pass, mpath_routing_pass, runs=5):
		super().__init__()

		self.sabre_pass = PassManager([
#			SabreLayout(coupling_map),
#			ApplyLayout(),
			SabreSwap(coupling_map, heuristic='decay'),
			Unroller(G_QISKIT_GATE_SET)
		])
		self.mpath_pass = PassManager([
#			mpath_mapping_pass,
#			ApplyLayout(),
			LayerViewPass(),
			mpath_routing_pass,
			Unroller(G_QISKIT_GATE_SET)
		])
		self.look_pass = PassManager([
#			SabreLayout(coupling_map, routing_pass=LookaheadSwap(coupling_map)),
#			ApplyLayout(),
			LookaheadSwap(coupling_map, search_depth=4, search_width=4),
			Unroller(G_QISKIT_GATE_SET)
		])
		self.runs = runs

		self.benchmark_results = None
	
	def run(self, dag):
		original_circuit = dag_to_circuit(dag)
		original_circuit_size = original_circuit.size()

		self.benchmark_results = {
			'SABRE CNOTs': 0.0,
			'MPATH CNOTs': 0.0,
			'LOOK CNOTs': 0.0,
			'SABRE Depth': 0.0,
			'MPATH Depth': 0.0,
			'LOOK Depth': 0.0,
			'SABRE Time': 0.0,
			'MPATH Time': 0.0,
			'LOOK Time': 0.0
		}

		sabre_swaps, mpath_swaps, look_swaps = 0, 0, 0
		sabre_depth, mpath_depth, look_depth = 0, 0, 0
		sabre_time, mpath_time, look_time = 0, 0, 0
		sabre_swaps, mpath_swaps, sabre_depth, mpath_depth, sabre_time, mpath_time = 0, 0, 0, 0, 0, 0
		for _ in range(self.runs):
			# Run dag on both passes. 
			# SABRE
			start = timer()
			sabre_circ = self.sabre_pass.run(original_circuit)
			end = timer()
			self.benchmark_results['SABRE CNOTs'] += (sabre_circ.size() - original_circuit_size) / self.runs
			self.benchmark_results['SABRE Depth'] += (sabre_circ.depth()) / self.runs
			self.benchmark_results['SABRE Time'] += (end - start) / self.runs
			# MPATH
			start = timer()
			mpath_circ = self.mpath_pass.run(original_circuit)
			end = timer()
			if self.benchmark_results['MPATH CNOTs'] < 0 or mpath_circ.size() < original_circuit_size:
				self.benchmark_results['MPATH CNOTs'] = -1.0
				self.benchmark_results['MPATH Depth'] = -1.0
				self.benchmark_results['MPATH Time'] = -1.0
			else:
				self.benchmark_results['MPATH CNOTs'] += (mpath_circ.size() - original_circuit_size) / self.runs
				self.benchmark_results['MPATH Depth'] += (mpath_circ.depth()) / self.runs
				self.benchmark_results['MPATH Time'] += (end - start) / self.runs
			# LOOK
#			try:
#				start = timer()
#				look_circ = self.look_pass.run(original_circuit)
#				end = timer()
#				if self.benchmark_results['LOOK CNOTs'] >= 0:
#					self.benchmark_results['LOOK CNOTs'] += (look_circ.size() - original_circuit_size) / self.runs
#					self.benchmark_results['LOOK Depth'] += (look_circ.depth()) / self.runs
#					self.benchmark_results['LOOK Time'] += (end - start) / self.runs
#			except TranspilerError:
#				self.benchmark_results['LOOK CNOTs'] = -1.0
#				self.benchmark_results['LOOK Depth'] = -1.0
#				self.benchmark_results['LOOK Time'] = -1.0

def b_qasmbench(coupling_map, dataset='medium', out_file='qasmbench.csv', max_swaps=10, max_lookahead=5, runs=5):
	sabre_routing_pass = SabreSwap(coupling_map)
	mpath_routing_pass = MultipathSwap(coupling_map, max_swaps=max_swaps, max_lookahead=max_lookahead)

	basis_pass = Unroller(G_QISKIT_GATE_SET)

	layout_passes = [
		SabreLayout(coupling_map),
		TrivialLayout(coupling_map)
	]

	layout_pass_names = ['SABRE', 'Trivial']

	data = {}
	for (i, mpath_mapping_pass) in enumerate(layout_passes):
		benchmark_pass = BenchmarkPass(mpath_mapping_pass, mpath_routing_pass, runs=runs)
		benchmark_pm = PassManager([
			basis_pass, 
			mpath_mapping_pass, 
			ApplyLayout(), 
			benchmark_pass
		]) 

		sub_data = {}
		if dataset == 'zulehner':
			benchmark_suite = zulehner
		elif dataset == 'medium':
			benchmark_suite = qasmbench_medium
		else:
			benchmark_suite = qasmbench_large

		used_benchmarks = []
		for qb_file in benchmark_suite:
			if dataset == 'zulehner':
				circ = QuantumCircuit.from_qasm_file('benchmarks/zulehner/%s' % qb_file)
			else:
				circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/%s/%s/%s.qasm' % (dataset, qb_file, qb_file))	
			if circ.depth() > 3000:
				continue
			used_benchmarks.append(qb_file)
			_pad_circuit_to_fit(circ, coupling_map)

			print('[%s]' % qb_file)

			benchmark_pm.run(circ)
			benchmark_results = benchmark_pass.benchmark_results 	
			if benchmark_results['SABRE CNOTs'] == -1:
				print('\tN/A')
			else:
				for x in benchmark_results:
					print('\t%s: %.3f' % (x, benchmark_results[x]))
			for x in benchmark_results:
				if x not in sub_data:
					sub_data[x] = []
				sub_data[x].append(benchmark_results[x])
		for x in sub_data:
			data['[%s] %s' % (layout_pass_names[i], x)] = sub_data[x] 
	df = pd.DataFrame(data=data, index=used_benchmarks)
	df.to_csv(out_file)
	
if __name__ == '__main__':
	mode = argv[1]
	
	coupling_style = argv[2]
	max_swaps = int(argv[3])
	max_look = int(argv[4])
	runs = int(argv[5])
	file_out = argv[6]

	print('Config:\n\tmode: %s\n\tcoupling style: %s\n\tmax swaps: %d\n\tmax lookahead: %d\n\truns: %d'
			% (mode, coupling_style, max_swaps, max_look, runs))

	if coupling_style == 'toronto':
		coupling_map = CouplingMap(ibm_toronto)
	elif mode == 'medium':
		if coupling_style == 'grid':
			coupling_map = CouplingMap.from_grid(3, 5)
		elif coupling_style == 'linear':
			coupling_map = CouplingMap.from_line(15)
		else:
			coupling_map = CouplingMap.from_ring(15)
	else:
		if coupling_style == 'grid':
			coupling_map = CouplingMap.from_grid(4, 7)
		elif coupling_style == 'linear':
			coupling_map = CouplingMap.from_line(28)
		else:
			coupling_map = CouplingMap.from_ring(28)
	b_qasmbench(coupling_map, dataset=mode, max_swaps=max_swaps, max_lookahead=max_look, runs=runs, out_file=file_out)
