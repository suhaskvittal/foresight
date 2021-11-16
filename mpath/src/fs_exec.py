"""
    author: Suhas Vittal
    date:   14 September 2021 @ 2:36 p.m. EST
"""

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit import Aer

import numpy as np

from fs_util import G_QISKIT_GATE_SET

from timeit import default_timer as timer
from sys import argv

SIMULATOR = Aer.get_backend('qasm_simulator')
DEFAULT_SHOTS = 8192

def draw(circ):
    print(circ.draw(output='text'))

def _pad_circuit_to_fit(circ, coupling_map):
    while circ.num_qubits < coupling_map.size():
        circ.add_bits([Qubit()])

def exec_sim(circ, shots=DEFAULT_SHOTS, basis_gates=G_QISKIT_GATE_SET, noise_model=None):
    if 'measure' not in circ.count_ops():
        circ.measure_active()
    job = SIMULATOR.run(
        circ,
        shots=shots,
        basis_gates=basis_gates,
        noise_model=noise_model
    )
    return job.result().get_counts(circ)
    
def total_variation_distance(counts1, counts2, shots=DEFAULT_SHOTS):
    calculated_set = set() 
    tvd = 0.0
    for x in counts1:
        calculated_set.add(x)
        if x not in counts2:
            tvd += np.abs(counts1[x])
        else:
            tvd += np.abs(counts1[x] - counts2[x])
    for x in counts2:
        if x in calculated_set:
            continue 
        if x not in counts1:
            tvd += np.abs(counts2[x])
        else:
            tvd += np.abs(counts1[x] - counts2[x])
    return tvd / shots
        
    
if __name__ == '__main__':
    circ_file = argv[1]
    circ = QuantumCircuit.from_qasm_file(circ_file)
    draw(circ)
