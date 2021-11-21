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
DEFAULT_SHOTS = 40000

def draw(circ):
    print(circ.draw(output='text'))

def _pad_circuit_to_fit(circ, coupling_map):
    while circ.num_qubits < coupling_map.size():
        circ.add_bits([Qubit()])

def _normalize_dict(d):
    eps = 1e-8
    f = 1.0/sum(d.values())
    for k in d:
        d[k] = d[k]*f
        if d[k] == 1:
            d[k] -= eps
    return d

def _2norm(x):
    if isinstance(x, list):
        s = 0
        for i in x:
            s += i**2
        return np.sqrt(s)
    else:
        return x

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
    p = _normalize_dict(counts1.copy())
    q = _normalize_dict(counts2.copy())
    eps = 1e-8

    for k in p.keys():
        if k not in q.keys():
            q[k] = eps
    for k in q.keys():
        if k not in p.keys():
            p[k] = eps
    p = _normalize_dict(p)
    q = _normalize_dict(q)
    qr = {x:q[x] for x in p.keys()}
    tvd = 0
    for x in p:
        tvd += np.abs(p[x] - qr[x])
    return tvd/2
    
if __name__ == '__main__':
    circ_file = argv[1]
    circ = QuantumCircuit.from_qasm_file(circ_file)
    draw(circ)
