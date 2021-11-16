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

from fs_layerview import LayerViewPass
from fs_exec import _pad_circuit_to_fit, draw
from fs_util import *
from fs_noise import * 
from fs_benchmark_pass import BenchmarkPass

from jkq import qmap

import pandas as pd
import pickle as pkl
import numpy as np

from sys import argv
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import tracemalloc
    
def benchmark(coupling_map, arch_file, dataset='medium', out_file='qasmbench.csv', runs=5, **kwargs):
    basis_pass = Unroller(G_QISKIT_GATE_SET)

    data = defaultdict(list)
    if 'vl' in dataset or kwargs['noisy']:
        compare = ['sabre', 'foresight']
    else:
        compare = ['sabre', 'foresight', 'ssonly']
    benchmark_pass = BenchmarkPass(coupling_map, runs=runs, compare=compare, compute_stats=False, **kwargs)
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
        benchmark_folder, benchmark_suite = G_ZULEHNER
    elif dataset == 'medium':
        benchmark_suite = G_QASMBENCH_MEDIUM
    elif dataset == 'large':
        benchmark_suite = G_QASMBENCH_LARGE
    elif dataset == 'qaoask':
        benchmark_folder, benchmark_suite = G_QAOA_SK
    elif dataset == 'qaoa3rl':
        benchmark_folder, benchmark_suite = G_QAOA_3RL
    elif dataset == 'qaoa3rvl':
        benchmark_folder, benchmark_suite = G_QAOA_3RVL
    elif dataset == 'bvvl':
        benchmark_folder, benchmark_suite = G_BV_VL

    used_benchmarks = []
    for qb_file in benchmark_suite:
        if dataset == 'medium' or dataset == 'large':
            circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/%s/%s/%s.qasm' % (dataset, qb_file, qb_file))    
        else:
            circ = QuantumCircuit.from_qasm_file('%s/%s' % (benchmark_folder, qb_file))
        if circ.depth() > 2000:
            continue
        used_benchmarks.append(qb_file)
        print('[%s]' % qb_file)
        _pad_circuit_to_fit(circ, coupling_map)
        circ = filter_pass.run(circ)

        benchmark_results = defaultdict(int)
        try:
            benchmark_pm.run(circ)
            benchmark_results = benchmark_pass.benchmark_results    
            circ.remove_final_measurements()

            # Collect results from A* search
            benchmark_results['A* CNOTs'] = 0  # init in case we skip A* or an error occurs
            benchmark_results['A* Depth'] = 0
            benchmark_results['A* Time'] = 0
            benchmark_results['A* Memory'] = 0
            if 'vl' not in dataset:
                print('\t\t(A* start.)')
                qmap_start = timer()
                tracemalloc.start(25)
                qmap_res = qmap.compile(
                    circ,
                    arch=arch_file,
                    method=qmap.Method.heuristic
                    #method=qmap.Method.exact
                )
                ss = tracemalloc.take_snapshot()
                qmap_end = timer()
                tracemalloc.stop()
                print('\t\t(A* done.)')
                # Benchmark A* search
                qmap_circ = QuantumCircuit.from_qasm_str(qmap_res['mapped_circuit']['qasm'])
                qmap_circ.global_phase = circ.global_phase
                qmap_circ = qmap_pass.run(qmap_circ)  # Compile gates to basis gates.
                benchmark_results['A* CNOTs'] = qmap_circ.count_ops()['cx']
                benchmark_results['A* Depth'] = qmap_circ.depth()
                benchmark_results['A* Time'] = qmap_end - qmap_start
                benchmark_results['A* Memory'] = sum(stat.size for stat in ss.statistics('traceback'))/1024.0
        except (QiskitError, KeyError) as error:
            print('\t\t(A* failure)')
            
        if benchmark_results['SABRE CNOTs'] == -1:
            print('\tN/A')
        else:
            for x in benchmark_results:
                print('\t%s: %.3f' % (x, benchmark_results[x]))
        for x in benchmark_results:
            data[x].append(benchmark_results[x])
    df = pd.DataFrame(data=data, index=used_benchmarks)
    df.to_csv(out_file)
    
if __name__ == '__main__':
    mode = argv[1]
    
    coupling_style = argv[2]
    runs = int(argv[3])
    file_out = argv[4]
    
    benchmark_kwargs = {
        'sim': True,
        'debug': False,
        'noisy': False,
        'mem': True
    }
    if len(argv) > 5:
        if '--nosim' in argv:
            benchmark_kwargs['sim'] = False
        if '--debug' in argv:
            benchmark_kwargs['debug'] = True
        if '--noisy' in argv:
            benchmark_kwargs['noisy'] = True
        if '--nomem' in argv:
            benchmark_kwargs['mem'] = False

    print('Config:\n\tmode: %s\n\tcoupling style: %s\n\truns: %d'
            % (mode, coupling_style, runs))

    if coupling_style == 'toronto':
        coupling_map = G_IBM_TORONTO 
        arch_file = 'arch/ibm_toronto.arch'  # For use with QMAP (Zulehner et al.)
    elif coupling_style == 'weber':
        coupling_map = G_GOOGLE_WEBER
        arch_file = 'arch/google_weber.arch'
        if benchmark_kwargs['noisy']:
            benchmark_kwargs['noise_model'] = google_weber_noise_model()
    elif coupling_style == 'aspen9':
        coupling_map = G_RIGETTI_ASPEN9
        arch_file = 'arch/rigetti_aspen9.arch'
    elif coupling_style == 'tokyo':
        coupling_map = G_IBM_TOKYO
        arch_file = 'arch/ibm_tokyo.arch'
    elif coupling_style == '100grid':
        coupling_map = G_100GRID
        arch_file = 'arch/100grid.arch'
    benchmark(coupling_map, arch_file, dataset=mode, runs=runs, out_file=file_out, **benchmark_kwargs)
