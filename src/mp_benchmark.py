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

from layerview import LayerViewPass
from mp_ips import MPATH_IPS
from mp_bsp import MPATH_BSP
from mp_exec import _bench_and_cmp, _pad_circuit_to_fit, draw
from mp_util import G_QISKIT_GATE_SET,\
					G_IBM_TORONTO,\
					G_GOOGLE_WEBER,\
					G_MPATH_IPS_LOOKAHEAD,\
					G_MPATH_IPS_SLACK,\
					G_MPATH_IPS_PATH_LIMIT,\
					G_MPATH_IPS_SOLN_CAP,\
					G_MPATH_BSP_TREE_WIDTH,\
					G_QASMBENCH_MEDIUM,\
					G_QASMBENCH_LARGE,\
					G_ZULEHNER,\
					G_QAOA

import pandas as pd
import pickle as pkl
import numpy as np

from sys import argv
from collections import defaultdict
from os import listdir
from os.path import isfile, join

class BenchmarkPass(AnalysisPass):
	def __init__(self, coupling_map, compare=['sabre', 'ips', 'bsp'], runs=5):
		super().__init__()

		self.benchmark_list = compare

		self.sabre_pass = PassManager([
			SabreSwap(coupling_map, heuristic='decay'),
			Unroller(G_QISKIT_GATE_SET)
		])
		self.ips_pass = PassManager([
			MPATH_IPS(
				coupling_map,
				slack=G_MPATH_IPS_SLACK,
				max_lookahead=G_MPATH_IPS_LOOKAHEAD,
				max_path_limit=G_MPATH_IPS_PATH_LIMIT,	
				solution_cap=G_MPATH_IPS_SOLN_CAP
			),
			Unroller(G_QISKIT_GATE_SET)
		])
		self.bsp_pass = PassManager([
			MPATH_BSP(
				coupling_map,
				tree_width_limit=G_MPATH_BSP_TREE_WIDTH
			),
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
		original_circuit_size = original_circuit.size()

		self.benchmark_results = defaultdict(float)

		for r in range(self.runs):
			# Run dag on both passes. 
			# SABRE
			if 'sabre' in self.benchmark_list:
				start = timer()
				sabre_circ = self.sabre_pass.run(original_circuit)
				end = timer()
				sabre_cnots = (sabre_circ.size() - original_circuit_size)
				if r == 0 or self.benchmark_results['SABRE CNOTs'] > sabre_cnots:
					self.benchmark_results['SABRE CNOTs'] = sabre_cnots
					self.benchmark_results['SABRE Depth'] = (sabre_circ.depth())
					self.benchmark_results['SABRE Time'] = (end - start)
			# MPATH IPS
			if 'ips' in self.benchmark_list:
				start = timer()
				ips_circ = self.ips_pass.run(original_circuit)
				end = timer()
				ips_cnots = (ips_circ.size() - original_circuit_size)
				if r == 0 or self.benchmark_results['MPATH_IPS CNOTs'] > ips_cnots:
					self.benchmark_results['MPATH_IPS CNOTs'] = ips_cnots
					self.benchmark_results['MPATH_IPS Depth'] = (ips_circ.depth())
					self.benchmark_results['MPATH_IPS Time'] = (end - start)
			# MPATH BSP
			if 'bsp' in self.benchmark_list:
				start = timer()
				bsp_circ = self.bsp_pass.run(original_circuit)
				end = timer()
				bsp_cnots = (bsp_circ.size() - original_circuit_size)
				if r == 0 or self.benchmark_results['MPATH_BSP CNOTs'] > bsp_cnots:
					self.benchmark_results['MPATH_BSP CNOTs'] = bsp_cnots
					self.benchmark_results['MPATH_BSP Depth'] = (bsp_circ.depth())
					self.benchmark_results['MPATH_BSP Time'] = (end - start)
			# LOOK
			if 'look' in self.benchmark_list:
				try:
					start = timer()
					look_circ = self.look_pass.run(original_circuit)
					end = timer()
					if self.benchmark_results['LOOK CNOTs'] >= 0:
						self.benchmark_results['LOOK CNOTs'] += (look_circ.size() - original_circuit_size) / self.runs
						self.benchmark_results['LOOK Depth'] += (look_circ.depth()) / self.runs
						self.benchmark_results['LOOK Time'] += (end - start) / self.runs
				except TranspilerError:
					self.benchmark_results['LOOK CNOTs'] = -1.0
					self.benchmark_results['LOOK Depth'] = -1.0
					self.benchmark_results['LOOK Time'] = -1.0
	
def b_qasmbench(coupling_map, dataset='medium', out_file='qasmbench.csv', runs=5):
	basis_pass = Unroller(G_QISKIT_GATE_SET)

	mapping_pass = SabreLayout(coupling_map, routing_pass=SabreSwap(coupling_map, heuristic='decay'))
	data = {}
	benchmark_pass = BenchmarkPass(coupling_map, runs=runs)
	benchmark_pm = PassManager([
		basis_pass, 
		mapping_pass, 
		ApplyLayout(), 
		benchmark_pass
	]) 

	if dataset == 'zulehner':
		benchmark_suite = G_ZULEHNER
	elif dataset == 'medium':
		benchmark_suite = G_QASMBENCH_MEDIUM
	elif dataset == 'large':
		benchmark_suite = G_QASMBENCH_LARGE
	elif dataset == 'qaoa':
		benchmark_suite = G_QAOA

	used_benchmarks = []
	for qb_file in benchmark_suite:
		if dataset == 'zulehner':
			circ = QuantumCircuit.from_qasm_file('benchmarks/zulehner/%s' % qb_file)
		elif dataset == 'qaoa':
			family, grid_type, circ = qb_file
			bench_name = '%s (%s)' % (family, grid_type)
		else:
			circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/%s/%s/%s.qasm' % (dataset, qb_file, qb_file))	
		if circ.depth() > 2000:
			continue
		if dataset == 'qaoa':
			used_benchmarks.append(bench_name)
			print('[%s]' % bench_name)
		else:
			used_benchmarks.append(qb_file)
			print('[%s]' % qb_file)
		_pad_circuit_to_fit(circ, coupling_map)

		benchmark_pm.run(circ)
		benchmark_results = benchmark_pass.benchmark_results 	
		if benchmark_results['SABRE CNOTs'] == -1:
			print('\tN/A')
		else:
			for x in benchmark_results:
				print('\t%s: %.3f' % (x, benchmark_results[x]))
		for x in benchmark_results:
			if x not in data:
				data[x] = []
			data[x].append(benchmark_results[x])
	df = pd.DataFrame(data=data, index=used_benchmarks)
	df.to_csv(out_file)
	
if __name__ == '__main__':
	mode = argv[1]
	
	coupling_style = argv[2]
	runs = int(argv[3])
	file_out = argv[4]

	print('Config:\n\tmode: %s\n\tcoupling style: %s\n\truns: %d'
			% (mode, coupling_style, runs))

	if coupling_style == 'toronto':
		coupling_map = G_IBM_TORONTO 
	elif coupling_style == 'weber':
		coupling_map = G_GOOGLE_WEBER
	b_qasmbench(coupling_map, dataset=mode, runs=runs, out_file=file_out)
