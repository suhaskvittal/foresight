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

from timeit import default_timer as timer
from copy import copy, deepcopy

from fs_layerview import LayerViewPass
from fs_exec import _pad_circuit_to_fit, draw
from fs_util import *
from fs_noise import * 
from fs_benchmark_pass import BenchmarkPass

import pandas as pd
import pickle as pkl
import numpy as np

from sys import argv
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import tracemalloc
import traceback
import pickle as pkl
    
def benchmark(coupling_map, arch_file, dataset='medium', out_file='qasmbench.csv', runs=5, **kwargs):
    data = defaultdict(list)
    if kwargs['noisy']:
        compare = ['sabre', 'foresight_dynamic']
    elif dataset == 'vqebench':
        compare = ['sabre', 'a*', 'foresight_dynamic','tket']
    elif dataset == 'zulehner':
        compare = ['sabre', 'a*','foresight_dynamic','tket']
    elif dataset == 'qasmbench_medium' or dataset == 'qasmbench_large': 
        compare = ['sabre', 'foresight_dynamic']
    else:
        compare = ['sabre', 'foresight_dynamic']
    benchmark_pass = BenchmarkPass(coupling_map, arch_file, runs=runs, compare=compare, compute_stats=False, **kwargs)
    benchmark_pm = PassManager([
        benchmark_pass
    ]) 

    if dataset == 'zulehner':
        benchmark_folder, benchmark_suite = G_ZULEHNER
    elif dataset == 'zulehner_partial':
        benchmark_folder, benchmark_suite = G_ZULEHNER_PARTIAL
    elif dataset == 'qasmbench_medium':
        benchmark_suite = G_QASMBENCH_MEDIUM
    elif dataset == 'qasmbench_large':
        benchmark_suite = G_QASMBENCH_LARGE
    elif dataset == 'qaoask':
        benchmark_folder, benchmark_suite = G_QAOA_SK
    elif dataset == 'qaoa3rl':
        benchmark_folder, benchmark_suite = G_QAOA_3RL
    elif dataset == 'qaoa3rvl':
        benchmark_folder, benchmark_suite = G_QAOA_3RVL
    elif dataset == 'bvl':
        benchmark_folder, benchmark_suite = G_BV_L
    elif dataset == 'bvvl':
        benchmark_folder, benchmark_suite = G_BV_VL
    elif dataset == 'vqebench':
        benchmark_folder, benchmark_suite = G_VQE
    elif dataset == 'qaoareal1':
        benchmark_folder, benchmark_suite = G_QAOA_REAL1
    elif dataset == 'qaoareal2':
        benchmark_folder, benchmark_suite = G_QAOA_REAL2
    elif dataset == 'qaoareal3':
        benchmark_folder, benchmark_suite = G_QAOA_REAL3

    used_benchmarks = []
    sim_counts = {}
    reached = False
    for qb_file in benchmark_suite:
        if dataset == 'qasmbench_medium':
            circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/medium/%s/%s.qasm' % (qb_file, qb_file))    
        elif dataset == 'qasmbench_large':
            circ = QuantumCircuit.from_qasm_file('benchmarks/qasmbench/large/%s/%s.qasm' % (qb_file, qb_file))    
        else:
            circ = QuantumCircuit.from_qasm_file('%s/%s' % (benchmark_folder, qb_file))
        if circ.depth() > 2500 and dataset == 'zulehner':
            continue
        n_qubits = circ.num_qubits
        used_benchmarks.append(qb_file)
        print('[%s]' % qb_file)
        _pad_circuit_to_fit(circ, coupling_map)
        if not kwargs['sim']:
            circ.remove_final_measurements()
        benchmark_pm.run(circ)
        benchmark_results = benchmark_pass.benchmark_results    
        benchmark_results['qubits'] = n_qubits
        if benchmark_results['SABRE CNOTs'] == -1:
            print('\tN/A')
        else:
            for x in benchmark_results:
                print('\t%s: %.3f' % (x, benchmark_results[x]))
        for x in benchmark_results:
            data[x].append(benchmark_results[x])
        if kwargs['sim']:
            sim_counts['qb_file'] = benchmark_pass.simulation_counts
    if kwargs['sim']:
        with open('counts_%s.pkl' % out_file, 'wb') as writer:
            pkl.dump(sim_counts, writer) 
    df = pd.DataFrame(data=data, index=used_benchmarks)
    df.to_csv(out_file)
    
if __name__ == '__main__':
    mode = 'medium'
    
    coupling_style = 'tokyo'
    runs = 1
    file_out = 'log'
    
    benchmark_kwargs = {
        'sim': True,
        'debug': False,
        'noisy': False,
        'mem': True,
        'slack': G_FORESIGHT_SLACK,
        'solncap': G_FORESIGHT_SOLN_CAP,
        'noise_factor': 1.0
    }
    
    if argv[1] == '-h':  # print out help
        print('Usage:')
        print('\t--nosim = do not simulate after routing.')
        print('\t--debug = print out debug messages.')
        print('\t--noisy = perform noisy routing (along with simulation if asked) -- only supported for Google Sycamore, Weber Architecture.')
        print('\t--nomem = do not measure memory usage (decreases time taken).')
        print('\t--dataset <d> where d is one of')
        print('\t\tzulehner (circuits used by Zulehner et al. in the A* paper)')
        print('\t\tzulehner_partial (a small selection of Zulehner et al.\'s circuits)')
        print('\t\tqasmbench_medium (a subset of the medium circuits from the QASMBENCH suite)')
        print('\t\tqasmbench_large (a subset of the large circuits from the QASMBENCH suite)')
        print('\t\tqaoask (Sherrington-Kirkpatrik QAOA circuits)')
        print('\t\tqaoa3rl (QAOA circuits for 3-regular graphs -- max size is 20 qubits)')
        print('\t\tqaoa3rvl (QAOA circuits for 3-regular graphs using around 100 qubits)')
        print('\t\tbvl (Bernstein-Vazirani circuits up to 50 qubits)')
        print('\t\tbvvl (Bernstein-Vazirani circuits up to 500 qubits)')
        print('\t\tvqebench (VQE Benchmarks from 4 to 8 qubits)')
        print('\t--runs <r> where r is the number of trials for each circuit')
        print('\t--coupling <backend> where backend is one of')
        print('\t\ttoronto (IBMQ Toronto -- 27 qubits)')
        print('\t\tweber (Google Sycamore, Weber Architecture -- 50 qubits)')
        print('\t\taspen9 (Rigetti Aspen9 -- 30 qubits)')
        print('\t\ttokyo (IBMQ Tokyo -- 20 qubits)')
        print('\t\t100grid (Grid of 100 qubits)')
        print('\t\t500grid (Grid of 500 qubits)')
        print('\t\t3heavyhex (IBM\'s presented sub-architecture for their Eagle QPU)')
        print('\t--output-file <file> where file is the name of the result file.')
        print('\t--slack <s> where s is the slack parameter used in ForeSight path identification.')
        print('\t--solncap <s> where s is the computation tree limit.')
        print('\t--noise-scale <f> where f is a factor by which to multiply noise effects.')
        exit()

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == '--nosim':
            benchmark_kwargs['sim'] = False
        elif arg == '--debug':
            benchmark_kwargs['debug'] = True
        elif arg == '--noisy':
            benchmark_kwargs['noisy'] = True
        elif arg == '--nomem':
            benchmark_kwargs['mem'] = False
        elif arg == '--dataset':
            mode = argv[i+1]
            i += 1
        elif arg == '--runs':
            runs = int(argv[i+1])  # get next argument
            i += 1
        elif arg == '--coupling':
            coupling_style = argv[i+1]
            i += 1
        elif arg == '--output-file':
            file_out = argv[i+1]
            i += 1 
        elif arg == '--slack':
            benchmark_kwargs['slack'] = float(argv[i+1])
            i += 1
        elif arg == '--solncap':
            benchmark_kwargs['solncap'] = int(argv[i+1])
        elif arg == '--noise-scale':
            benchmark_kwargs['noise_factor'] = float(argv[i+1])
            i += 1
        i += 1
    print('Config:\n\tdataset: %s\n\tcoupling style: %s\n\truns: %d\n\toutput-file: %s'
            % (mode, coupling_style, runs, file_out))
    for x in benchmark_kwargs:
        print('\t%s: %s' % (x, str(benchmark_kwargs[x])))

    if coupling_style == 'toronto':
        coupling_map = G_IBM_TORONTO 
        arch_file = 'arch/ibm_toronto.arch'  # For use with QMAP (Zulehner et al.)
    elif coupling_style == 'weber':
        coupling_map = G_GOOGLE_WEBER
        arch_file = 'arch/google_weber.arch'
        if benchmark_kwargs['noisy']:
            benchmark_kwargs['noise_model'] = google_weber_noise_model(benchmark_kwargs['noise_factor'])
    elif coupling_style == 'aspen9':
        coupling_map = G_RIGETTI_ASPEN9
        arch_file = 'arch/rigetti_aspen9.arch'
    elif coupling_style == 'tokyo':
        coupling_map = G_IBM_TOKYO
        arch_file = 'arch/ibm_tokyo.arch'
    elif coupling_style == '100grid':
        coupling_map = G_100GRID
        arch_file = 'arch/100grid.arch'
    elif coupling_style == '3heavyhex':
        coupling_map = G_IBM_3HEAVYHEX
        arch_file = 'arch/ibm_3heavyhex.arch'
    elif coupling_style == '500grid':
        coupling_map = G_500GRID
        arch_file = 'arch/500grid.arch'
    benchmark(coupling_map, arch_file, dataset=mode, runs=runs, out_file=file_out, **benchmark_kwargs)
