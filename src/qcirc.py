"""
	author: Suhas Vittal
	date: 	14 September 2021 @ 2:36 p.m. EST
"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit import Aer

from csolvswap import ConvexSolverSwap
from mpathswap import MultipathSwap

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

def _bench_and_cmp(ref_circ, coupling_map, pm1, pm2, runs=100):
	swaps, max_swaps, min_swaps = 0.0, None, None
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
		# Update sats
		s = (circ1.size() - circ2.size())
		swaps += s / runs
		mean_t1 += time1 / runs
		mean_t2 += time2 / runs
		if max_swaps is None:
			max_swaps = s
			min_swaps = s
		else:
			if s > max_swaps:
				max_swaps = s
			elif s < min_swaps:
				min_swaps = s
	if runs == 1:
		for circ in [ref_circ, circ1, circ2]:
			res = BACKEND.run(circ, shots=1024).result()
			print(res.get_counts(circ))	
			draw(circ)
	return swaps, max_swaps, min_swaps, mean_t1, mean_t2

if __name__ == '__main__':
	n, m = 2, 5
	s = 7
	circ = QuantumCircuit.from_qasm_file('benchmarks/BV-10.qasm') 
	coupling_map = CouplingMap.from_grid(n, m)

	basis_pass = Unroller(G_QISKIT_GATE_SET)
	trivial_layout_pass = TrivialLayout(coupling_map)
	gate1q_pass = Optimize1qGates()
	gatecx_pass = CXCancellation()

	sabre_routing_pass = SabreSwap(coupling_map)
	sabre_mapping_pass = SabreLayout(coupling_map, routing_pass=None)
	csolv_routing_pass = MultipathSwap(coupling_map, max_swaps=s)

	ipass_list = [basis_pass] 
	fpass_list = [gate1q_pass, gatecx_pass]

	pass_list1, pass_list2 = ipass_list.copy(), ipass_list.copy()
	pass_list1.append(sabre_mapping_pass)
	pass_list1.append(sabre_routing_pass)
	pass_list2.append(csolv_routing_pass)

	pass_list1.extend(fpass_list)
	pass_list2.extend(fpass_list)

	pm1 = PassManager(pass_list1)
	pm2 = PassManager(pass_list2)
	print(_bench_and_cmp(circ, coupling_map, pm1, pm2, runs=int(argv[1])))
