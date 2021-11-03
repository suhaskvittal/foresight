"""
    author: Suhas Vittal
    date:   21 October 2021
"""
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.library import QuantumVolume
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import *

import numpy as np
import pandas as pd

from mp_benchmark_pass import BenchmarkPass
from mp_util import G_IBM_TORONTO,\
                    G_RIGETTI_ASPEN9,\
                    G_GOOGLE_WEBER,\
                    G_QISKIT_GATE_SET
from mp_exec import _pad_circuit_to_fit

from copy import copy
from sys import argv
from collections import defaultdict

def profile_optimal_workload(out_file, coupling_map, runs=1000):
    data = defaultdict(list)    

    benchmark_pass = BenchmarkPass(coupling_map, None, compare=['sabre', 'ips'], runs=1)
    profile_pm = PassManager([
        Unroller(G_QISKIT_GATE_SET),
        SabreLayout(coupling_map, routing_pass=SabreSwap(coupling_map, heuristic='decay')),
        ApplyLayout(),
        benchmark_pass
    ])

    for r in range(runs):
        circ = _generate_random_circuit(coupling_map.size(), np.random.randint(50, high=200))
        _pad_circuit_to_fit(circ, coupling_map)
        print('run %d' % r)
        data['size'].append(circ.size())
        data['depth'].append(circ.depth())

        print('\trunning profiler...')
        profile_pm.run(circ)
        benchmark_results = benchmark_pass.benchmark_results
        for x in benchmark_results:
            data[x].append(benchmark_results[x])
        # Measure improvement
        # Logging
        for x in data:
            print('\t%s: %.3f' % (x, data[x][-1]))
    df = pd.DataFrame(data=data)
    df.to_csv(out_file)

def _generate_random_circuit(num_qubits, num_ops):
    circ = QuantumCircuit(num_qubits)   
    depth_probability = 0.15*np.random.rand() + 0.85
    i = 0
    while i < num_ops:
        # Random h, x, y, z, cx, cy, cz, t operations
        rand_op_number = np.random.randint(0, high=8)
        q1 = np.random.randint(0, high=num_qubits)
        if i == 0 or np.random.rand() >= depth_probability:
            q2 = np.random.randint(0, high=num_qubits)
        if q2 == q1:
            q2 = (q1 + 1) % num_qubits
        # Apply op.
        if rand_op_number == 0:
            circ.h(q1)
        elif rand_op_number == 1:
            circ.x(q1)
        elif rand_op_number == 2:
            circ.y(q1)
        elif rand_op_number == 3:
            circ.z(q1)
        elif rand_op_number == 4:
            circ.cx(q1, q2)
            i += 1
        elif rand_op_number == 5:
            circ.cy(q1, q2)
            i += 1
        elif rand_op_number == 6:
            circ.cz(q1, q2)
            i += 1
        elif rand_op_number == 7:
            circ.t(q1)
    return circ
                
if __name__ == '__main__':
    out_file = argv[1]
    coupling_map_name = argv[2]

    if coupling_map_name == 'toronto':
        coupling_map = G_IBM_TORONTO
    elif coupling_map_name == 'aspen9':
        coupling_map = G_RIGETTI_ASPEN9
    elif coupling_map_name == 'weber':
        coupling_map = G_GOOGLE_WEBER
    profile_optimal_workload(out_file, coupling_map)
                
