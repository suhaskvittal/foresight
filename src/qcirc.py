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

from mpathswap import MultipathSwap, MPSWAP_ERRNO

from timeit import default_timer as timer

from sys import argv

G_QISKIT_OPT_NONE = 0
G_QISKIT_OPT_LIGHT = 1
G_QISKIT_OPT_HEAVY = 2
G_QISKIT_OPT_HEAVIER = 3

G_QISKIT_OPT_LVL = G_QISKIT_OPT_HEAVY  
G_QISKIT_GATE_SET = ['u1', 'u2', 'u3', 'cx']

BACKEND = Aer.get_backend('qasm_simulator')

def draw(circ):
	print(circ.draw(output='text'))

def _build_pass_manager(router, coupling_map):
	basis_pass = Unroller(G_QISKIT_GATE_SET)
	apply_layout_pass = ApplyLayout()
	gate1q_pass = Optimize1qGates()

	sabre_mapping_pass = SabreLayout(coupling_map, routing_pass=None)
	
	if router is None:
		return PassManager([basis_pass, gate1q_pass])
	else:
		return PassManager([basis_pass, sabre_mapping_pass, apply_layout_pass, router, gate1q_pass]) 

def _pad_circuit_to_fit(circ, coupling_map):
	while circ.num_qubits < coupling_map.size():
		circ.add_bits([Qubit()])

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
		s = (circ1.size() - circ2.size())
		mean_first_swaps += circ1.size() / runs
		mean_second_swaps += circ2.size() / runs
		mean_depth1 += circ1.depth() / runs
		mean_depth2 += circ2.depth() / runs
		mean_t1 += time1 / runs
		mean_t2 += time2 / runs
	if runs == 1 and show:
		for circ in [circ0, circ1, circ2]:
			draw(circ)
	return mean_first_swaps, mean_second_swaps, mean_depth1, mean_depth2, mean_t1, mean_t2

if __name__ == '__main__':
	n, m = int(argv[1]), int(argv[2])
	s = max(n+m, 4)
	circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/%s/%s/%s.qasm' % (argv[3], argv[4], argv[4])) 
	coupling_map = CouplingMap.from_grid(n, m)
	#coupling_map = CouplingMap.from_line(n)
	#coupling_map = CouplingMap.from_ring(n)

	_pad_circuit_to_fit(circ, coupling_map)

	sabre_routing_pass = SabreSwap(coupling_map)
	mpath_routing_pass = MultipathSwap(coupling_map, max_swaps=s)

	pm0 = _build_pass_manager(None, coupling_map) 
	pm1 = _build_pass_manager(sabre_routing_pass, coupling_map)
	pm2 = _build_pass_manager(mpath_routing_pass, coupling_map)

	print(_bench_and_cmp(circ, coupling_map, pm0, pm1, pm2, runs=int(argv[5]), show=True))

