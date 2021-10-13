"""
	author: Suhas Vittal
	date:	5 October 2021 @ 2:35 p.m. EST
"""

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit import Aer

from timeit import default_timer as timer
from copy import copy, deepcopy

from mpathswap import MultipathSwap
from layerview import LayerViewPass
from qcirc import _bench_and_cmp, _build_pass_manager, _pad_circuit_to_fit

import pandas as pd
import numpy as np

from sys import argv

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

class BenchmarkPass(AnalysisPass):
	def __init__(self, sabre_routing_pass, mpath_routing_pass, runs=5):
		super().__init__()

		self.sabre_routing_pass = sabre_routing_pass
		self.mpath_routing_pass = mpath_routing_pass
		self.runs = runs

		self.benchmark_results = None
	
	def run(self, dag):
		original_circuit_size = dag.size()

		primary_layer_view, secondary_layer_view = self.property_set['primary_layer_view'], self.property_set['secondary_layer_view']
		basis_pass = Unroller(G_QISKIT_GATE_SET)

		sabre_swaps, mpath_swaps, sabre_depth, mpath_depth, sabre_time, mpath_time = 0, 0, 0, 0, 0, self.property_set['bench_layer_view']
		for _ in range(self.runs):
			plv_cpy, slv_cpy = deepcopy(primary_layer_view), deepcopy(secondary_layer_view)
			# Run dag on both passes. 
			# SABRE
			start = timer()
			sabre_dag = self.sabre_routing_pass.run(dag)
			end = timer()
			sabre_time += (end - start) / self.runs
			# MPATH
			start = timer()
			mpath_dag = self.mpath_routing_pass.run(dag, primary_layer_view=plv_cpy, secondary_layer_view=slv_cpy)
			end = timer()
			mpath_time += (end - start) / self.runs
			# Unroll SWAPs to CNOTs 
			sabre_dag, mpath_dag = basis_pass.run(sabre_dag), basis_pass.run(mpath_dag)
			# Compare dags.
			if mpath_dag.size() < original_circuit_size:
				sabre_swaps, mpath_swaps, sabre_depth, mpath_depth, sabre_time, mpath_time = -1, -1, -1, -1, -1, -1
				break
			sabre_swaps += (sabre_dag.size() - original_circuit_size) / self.runs
			mpath_swaps += (mpath_dag.size() - original_circuit_size) / self.runs
			sabre_depth += (sabre_dag.depth()) / self.runs
			mpath_depth += (mpath_dag.depth()) / self.runs
		self.benchmark_results = [sabre_swaps, mpath_swaps, sabre_depth, mpath_depth, sabre_time, mpath_time]

def b_qasmbench(coupling_map, dataset='medium', out_file='qasmbench.csv', max_swaps=10, max_lookahead=5, runs=5):
	sabre_routing_pass = SabreSwap(coupling_map)
	mpath_routing_pass = MultipathSwap(coupling_map, max_swaps=max_swaps, max_lookahead=max_lookahead)

	basis_pass = Unroller(G_QISKIT_GATE_SET)
	apply_layout_pass = ApplyLayout()
	sabre_mapping_pass = SabreLayout(coupling_map, routing_pass=None)
	layer_view_pass = LayerViewPass()

	benchmark_pass = BenchmarkPass(sabre_routing_pass, mpath_routing_pass, runs=runs)

	benchmark_pm = PassManager([basis_pass, sabre_mapping_pass, apply_layout_pass, layer_view_pass, benchmark_pass]) 
		
	data = {
		'SABRE Swaps': [],
		'MPATH Swaps': [],
		'SABRE Depth': [],
		'MPATH Depth': [],
		'SABRE Time': [],
		'MPATH Time': []
	}

	benchmark_suite = qasmbench_medium if dataset=='medium' else qasmbench_large

	for qb_file in benchmark_suite:
		circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/%s/%s/%s.qasm' % (dataset, qb_file, qb_file))	
		_pad_circuit_to_fit(circ, coupling_map)

		benchmark_pm.run(circ)
		sabre_swaps, mpath_swaps, sabre_depth, mpath_depth, sabre_time, mpath_time = benchmark_pass.benchmark_results 	
		if sabre_swaps == -1:
			mpath_swaps = 'N/A'
			mpath_depth = 'N/A'
			mpath_time = 'N/A'
			print('[%s]\n\tN/A' % qb_file)
		else:
			print('[%s]\n\tSABRE Swaps: %.3f\n\tMultipath Swaps: %.3f\n\tSABRE Depth: %.3f\n\tMultipath Depth: %.3f\n\tSABRE Time: %.3f\n\tMultipath Time: %.3f'
					% (qb_file, sabre_swaps, mpath_swaps, sabre_depth, mpath_depth, sabre_time, mpath_time))
		data['SABRE Swaps'].append(sabre_swaps)
		data['MPATH Swaps'].append(mpath_swaps)
		data['SABRE Depth'].append(sabre_depth)
		data['MPATH Depth'].append(mpath_depth)
		data['SABRE Time'].append(sabre_time)
		data['MPATH Time'].append(mpath_time)
	
	df = pd.DataFrame(data=data, index=benchmark_suite)
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
