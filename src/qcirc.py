"""
	author: Suhas Vittal
	date: 	14 September 2021 @ 2:36 p.m. EST
"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile

from csolvswap import ConvexSolverSwap

G_QISKIT_OPT_NONE = 0
G_QISKIT_OPT_LIGHT = 1
G_QISKIT_OPT_HEAVY = 2
G_QISKIT_OPT_HEAVIER = 3

G_QISKIT_OPT_LVL = G_QISKIT_OPT_HEAVY  
G_QISKIT_GATE_SET = ['u1', 'u2', 'u3', 'cx']

def draw(circ):
	print(circ.draw(output='text'))

def _bench_and_cmp(ref_circ, coupling_map, pm1, pm2, runs=100):
	avg = 0.0
	for _ in range(runs):
		circ1 = pm1.run(ref_circ)
		circ2 = pm2.run(ref_circ)
		avg += (circ1.size() - circ2.size()) / runs
	if runs == 1:
		draw(circ1)
		draw(circ2)
	return avg

if __name__ == '__main__':
	circ = QuantumCircuit.from_qasm_file('benchmarks/BV-10.qasm') 
	coupling_map = CouplingMap.from_grid(2, 5)

	basis_pass = Unroller(G_QISKIT_GATE_SET)
	trivial_layout_pass = TrivialLayout(coupling_map)

	sabre_routing_pass = SabreSwap(coupling_map)
	csolv_routing_pass = ConvexSolverSwap(coupling_map, max_swaps=5)

	ipass_list = [basis_pass, trivial_layout_pass] 
	fpass_list = [basis_pass]

	pass_list1, pass_list2 = ipass_list.copy(), ipass_list.copy()
	pass_list1.append(sabre_routing_pass)
	pass_list2.append(csolv_routing_pass)

	pass_list1.extend(fpass_list)
	pass_list2.extend(fpass_list)

	pm1 = PassManager(pass_list1)
	pm2 = PassManager(pass_list2)
	print(_bench_and_cmp(circ, coupling_map, pm1, pm2, runs=10))
