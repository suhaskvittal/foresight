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

from mp_layerview import LayerViewPass
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
    
if __name__ == '__main__':
    circ_file = argv[1]
    circ = QuantumCircuit.from_qasm_file(circ_file)
    draw(circ)
