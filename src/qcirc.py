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

from mpathswap import MultipathSwap
from layerview import LayerViewPass

from timeit import default_timer as timer

from sys import argv

import numpy as np

G_QISKIT_OPT_NONE = 0
G_QISKIT_OPT_LIGHT = 1
G_QISKIT_OPT_HEAVY = 2
G_QISKIT_OPT_HEAVIER = 3

G_QISKIT_OPT_LVL = G_QISKIT_OPT_HEAVY  
G_QISKIT_GATE_SET = ['u1', 'u2', 'u3', 'cx']

BACKEND = Aer.get_backend('qasm_simulator')

ibm_toronto = np.array([[0, 1], [1, 0], [1, 2], [1, 4], [2, 1], [2, 3], [3, 2], [3, 5], [4, 1], [4, 7], [5, 3], [5, 8], [6, 7], [7, 4], [7, 6], [7, 10], [8, 5], [8, 9], [8, 11], [9, 8], [10, 7], [10, 12], [11, 8], [11, 14], [12, 10], [12, 13], [12, 15], [13, 12], [13, 14], [14, 11], [14, 13], [14, 16], [15, 12], [15, 18], [16, 14], [16, 19], [17, 18], [18, 15], [18, 17], [18, 21], [19, 16], [19, 20], [19, 22], [20, 19], [21, 18], [21, 23], [22, 19], [22, 25], [23, 21], [23, 24], [24, 23], [24, 25], [25, 22], [25, 24], [25, 26], [26, 25]])


def draw(circ):
	print(circ.draw(output='text'))

def _build_pass_manager(router, coupling_map, mapping='sabre'):
	basis_pass = Unroller(G_QISKIT_GATE_SET)
	apply_layout_pass = ApplyLayout()
	gate1q_pass = Optimize1qGates()
	layer_view = LayerViewPass()

	if mapping == 'sabre':
		layout_pass = SabreLayout(coupling_map)
	else:
		layout_pass = TrivialLayout(coupling_map)
	
	if router is None:
		return PassManager([basis_pass, gate1q_pass])
	else:
		return PassManager([basis_pass, layout_pass, apply_layout_pass, layer_view, router]) 

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
#	if runs == 1 and show:
#		base_counts = BACKEND.run(ref_circ, shots=1024).result().get_counts(ref_circ)
#		sabre_counts = BACKEND.run(circ1, shots=1024).result().get_counts(circ1)
#		mpath_counts = BACKEND.run(circ2, shots=1024).result().get_counts(circ2)
#		draw(ref_circ)
		draw(circ0)
		draw(circ1)
		draw(circ2)
#		print('DIFF: [SABRE] %.3f, [MPATH] %.3f' % (_diff(base_counts, sabre_counts), _diff(base_counts, mpath_counts)))
		
	return mean_first_swaps, mean_second_swaps, mean_depth1, mean_depth2, mean_t1, mean_t2

if __name__ == '__main__':
	n, m = int(argv[1]), int(argv[2])
	s = max(n+m, 4)
	circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/%s/%s/%s.qasm' % (argv[3], argv[4], argv[4])) 
#	circ = QuantumCircuit.from_qasm_file('benchmarks/BV-10.qasm')
#	circ = QuantumCircuit.from_qasm_file('benchmarks/cnot_test_n5.qasm')

	coupling_map = CouplingMap.from_grid(n, m)
#	coupling_map = CouplingMap(ibm_toronto)
	#coupling_map = CouplingMap.from_line(n)
	#coupling_map = CouplingMap.from_ring(n)

	_pad_circuit_to_fit(circ, coupling_map)

	sabre_routing_pass = SabreSwap(coupling_map, heuristic='lookahead')
	mpath_routing_pass = MultipathSwap(coupling_map, max_swaps=s, max_lookahead=8, solution_cap=8)

	pm0 = _build_pass_manager(None, coupling_map) 
	pm1 = _build_pass_manager(sabre_routing_pass, coupling_map, mapping='trivial')
	pm2 = _build_pass_manager(mpath_routing_pass, coupling_map, mapping='trivial')

	print(_bench_and_cmp(circ, coupling_map, pm0, pm1, pm2, runs=int(argv[5]), show=True))

