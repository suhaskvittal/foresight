"""
    author: Suhas Vittal
    date:   5 October 2021 @ 2:35 p.m. EST
"""

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import *
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit.exceptions import QiskitError

from timeit import default_timer as timer
from copy import copy, deepcopy

from mp_layerview import LayerViewPass
from mp_ips import MPATH_IPS
from mp_bsp import MPATH_BSP
from mp_exec import _bench_and_cmp, _pad_circuit_to_fit, draw
from mp_util import G_QISKIT_GATE_SET,\
                    G_IBM_TORONTO,\
                    G_GOOGLE_WEBER,\
                    G_RIGETTI_ASPEN9,\
                    G_IBM_TOKYO,\
                    G_QASMBENCH_MEDIUM,\
                    G_QASMBENCH_LARGE,\
                    G_MPATH_IPS_SLACK,\
                    G_MPATH_IPS_SOLN_CAP,\
                    G_ZULEHNER,\
                    G_QAOA
from mp_benchmark_pass import BenchmarkPass

from jkq import qmap

import pandas as pd
import pickle as pkl
import numpy as np

from sys import argv
from collections import defaultdict
from os import listdir
from os.path import isfile, join
    
def b_qasmbench(coupling_map, arch_file, hybrid_data_file, dataset='medium', out_file='qasmbench.csv', runs=5):
    basis_pass = Unroller(G_QISKIT_GATE_SET)

    data = {}
    if dataset == 'qaoa':
        compare = ['sabre', 'ips', 'ssonly']
    else:
        compare = ['sabre', 'ips', 'ssonly']
    benchmark_pass = BenchmarkPass(coupling_map, hybrid_data_file, runs=runs, compare=compare, compute_stats=False)
    benchmark_pm = PassManager([
        basis_pass, 
        benchmark_pass
    ]) 
    qmap_pass = PassManager([
        Unroller(['u1', 'u2', 'u3', 'p', 'cx'])
    ])
    filter_pass = PassManager([
        basis_pass,
        TrivialLayout(coupling_map),
        ApplyLayout()
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
        if 'hybrid' in compare and circ.depth() > 200:
            continue
        if circ.depth() > 2000 or circ.depth() < 30:
            continue
        if dataset == 'qaoa':
            used_benchmarks.append(bench_name)
            print('[%s]' % bench_name)
        else:
            used_benchmarks.append(qb_file)
            print('[%s]' % qb_file)
        _pad_circuit_to_fit(circ, coupling_map)
        circ = filter_pass.run(circ)
        circ.remove_final_measurements()

        try:
            benchmark_pm.run(circ)
            benchmark_results = benchmark_pass.benchmark_results    

            # Collect results from A* search
            if dataset == 'zulehner' or dataset == 'qaoa':
                qmap_start = timer()
                qmap_res = qmap.compile(
                    circ,
                    arch=arch_file,
                    method=qmap.Method.heuristic
                )
                qmap_end = timer()
                # Benchmark A* search
                qmap_circ = QuantumCircuit.from_qasm_str(qmap_res['mapped_circuit']['qasm'])
                qmap_circ.global_phase = circ.global_phase
                qmap_circ = qmap_pass.run(qmap_circ)  # Compile gates to basis gates.
                benchmark_results['A* CNOTs'] = 3*qmap_res['mapped_circuit']['swaps']
                benchmark_results['A* Depth'] = qmap_circ.depth()
                benchmark_results['A* Time'] = qmap_end - qmap_start
            
            if benchmark_results['SABRE CNOTs'] == -1:
                print('\tN/A')
            else:
                for x in benchmark_results:
                    print('\t%s: %.3f' % (x, benchmark_results[x]))
            for x in benchmark_results:
                if x not in data:
                    data[x] = []
                data[x].append(benchmark_results[x])
        except QiskitError as error:
            print(error)
            used_benchmarks.pop()
            continue
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
        arch_file = 'arch/ibm_toronto.arch'  # For use with QMAP (Zulehner et al.)
        hybrid_data_file = 'profiles/toronto_profile.csv'
    elif coupling_style == 'weber':
        coupling_map = G_GOOGLE_WEBER
        arch_file = 'arch/google_weber.arch'
        hybrid_data_file = 'profiles/weber_profile.csv'
    elif coupling_style == 'aspen9':
        coupling_map = G_RIGETTI_ASPEN9
        arch_file = 'arch/rigetti_aspen9.arch'
        hybrid_data_file = 'profiles/aspen9_profile.csv'
    elif coupling_style == 'tokyo':
        coupling_map = G_IBM_TOKYO
        arch_file = 'arch/ibm_tokyo.arch'
        hybrid_data_file = ''  # undefined
    b_qasmbench(coupling_map, arch_file, hybrid_data_file, dataset=mode, runs=runs, out_file=file_out)
