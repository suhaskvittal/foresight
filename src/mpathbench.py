"""
	author: Suhas Vittal
	date:	5 October 2021 @ 2:35 p.m. EST
"""

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit import Aer

from mpathswap import MultipathSwap, MPSWAP_ERRNO
from qcirc import _bench_and_cmp, _build_pass_manager, _pad_circuit_to_fit

import pandas as pd

from sys import argv

qasmbench_medium = [
	'adder_n10',		# single adder
	'cc_n12',			# counterfeit coin
	'multiplier_n15', 	# binary multiplier
	'qf21_n15', 		# quantum phase estimation, factor 21			
	'sat_n11',		
	'dnn_n8',			# quantum deep neural net
	'seca_n11',			# shor's error correction
	'bv_n14',			# bernstein-vazirani algorithm 
	'ising_n10',		# ising gate sim
	'qaoa_n6',			
	'qpe_n9',			# quantum phase estimation
	'simon_n6'			# simon's algorithm	
]

qasmbench_large = [
	'bigadder_n18',		# ripple carry adder	
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


def b_qasmbench(coupling_map, dataset='medium'):
	sabre_routing_pass = SabreSwap(coupling_map)
	mpath_routing_pass = MultipathSwap(coupling_map, max_swaps=10)

	vanilla_pm = _build_pass_manager(None, coupling_map)
	sabre_pm = _build_pass_manager(sabre_routing_pass, coupling_map)
	mpath_pm = _build_pass_manager(mpath_routing_pass, coupling_map)
		
	data = {
		'Swap Diff': [],
		'SABRE Swaps': [],
		'MPATH Swaps': [],
		'SABRE Time': [],
		'MPATH Time': []
	}

	benchmark_suite = qasmbench_medium if dataset=='medium' else qasmbench_large

	for qb_file in benchmark_suite:
		circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/%s/%s/%s.qasm' % (dataset, qb_file, qb_file))	
		_pad_circuit_to_fit(circ, coupling_map)

		swap_diff, sabre_swaps, mpath_swaps, sabre_time, mpath_time = _bench_and_cmp(circ, coupling_map, vanilla_pm, sabre_pm, mpath_pm, runs=1)	
		if sabre_swaps == -1:
			swap_diff = 'N/A'
			mpath_swaps = 'N/A'
			mpath_time = 'N/A'
		print('[%s]\n\tSwap Diff: %.3f\n\tSABRE Swaps: %.3f\n\tMultipath Swaps: %.3f\n\tSABRE Time: %.3f\n\tMultipath Time: %.3f'
				% (qb_file, swap_diff, sabre_swaps, mpath_swaps, sabre_time, mpath_time))
		data['Swap Diff'].append(swap_diff)
		data['SABRE Swaps'].append(sabre_swaps)
		data['MPATH Swaps'].append(mpath_swaps)
		data['SABRE Time'].append(sabre_time)
		data['MPATH Time'].append(mpath_time)
	
	df = pd.DataFrame(data=data, index=benchmark_suite)
	df.to_csv('qasmbench_%s.csv' % dataset)
	
if __name__ == '__main__':
	mode = argv[1]
	if mode == 'medium':
		coupling_map = CouplingMap.from_grid(3, 5)
	else:
		coupling_map = CouplingMap.from_grid(4, 7)
	b_qasmbench(coupling_map, dataset=mode)
