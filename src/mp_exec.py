"""
	author: Suhas Vittal
	date: 	14 September 2021 @ 2:36 p.m. EST
"""

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit import Aer

import numpy as np

from layerview import LayerViewPass
from mp_bsp import MPATH_BSP
from mp_ips import MPATH_IPS
from mp_util import G_QISKIT_GATE_SET,\
					G_IBM_TORONTO

from timeit import default_timer as timer
from sys import argv

BACKEND = Aer.get_backend('qasm_simulator')

def draw(circ):
	print(circ.draw(output='text'))

def _pad_circuit_to_fit(circ, coupling_map):
	while circ.num_qubits < coupling_map.size():
		circ.add_bits([Qubit()])
	
def _diff(counts1, counts2):
	visited = set()
	diff = 0.0
	for x in counts1:
		diff += abs(counts1[x] - counts2[x]) if x in counts2 else 0.0
		visited.add(x)
	for x in counts2:
		if x not in visited:
			diff += counts2[x]
	return diff

def _bench_and_cmp(ref_circ, coupling_map, pm0, pm1, pm2, runs=100, show=False):
	circ0 = pm0.run(ref_circ)

	mean_first_swaps, mean_second_swaps = -circ0.size(), -circ0.size()
	mean_depth1, mean_depth2 = 0.0, 0.0
	mean_t1, mean_t2 = 0.0, 0.0
	for _ in range(runs):
		# Benchmark first pass.
		start = timer()
		circ1 = pm1.run(ref_circ)
		end = timer()
		time1 = end - start
		# Benchmark second pass.
		start = timer()
		circ2 = pm2.run(ref_circ)
		end = timer()
		time2 = end - start
		# Update stats
		if circ2.size() == 0:
			return -1, -1, -1, -1, -1, -1
		mean_first_swaps += circ1.size() / runs
		mean_second_swaps += circ2.size() / runs
		mean_depth1 += circ1.depth() / runs
		mean_depth2 += circ2.depth() / runs
		mean_t1 += time1 / runs
		mean_t2 += time2 / runs
	return mean_first_swaps, mean_second_swaps, mean_depth1, mean_depth2, mean_t1, mean_t2

if __name__ == '__main__':
	circ_file = argv[1]
	circ = QuantumCircuit.from_qasm_file(circ_file)
	draw(circ)
