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
from qiskit.converters import circuit_to_dag, dag_to_circuit

from timeit import default_timer as timer
from copy import copy, deepcopy

from fs_statsabre import *
from fs_foresight import *
from fs_util import *

import mqt.qmap

import pandas as pd
import pickle as pkl
import numpy as np

from sys import argv
from collections import defaultdict

import os
import time
import tracemalloc
import traceback
import pickle

BENCHMARK_PATH = '../benchmarks'
RUNS=5

### BASE LAYOUT PASS ###

def qiskitopt3_layout_pass(coupling_map, routing_pass=None, do_unroll=True):
    _unroll3q = [
        UnitarySynthesis(
            G_QISKIT_GATE_SET,
#            approximation_degree=1,
            min_qubits=3
        ),
        Unroll3qOrMore()
    ]

    _reset = [RemoveResetInZeroState()]
    _meas = [OptimizeSwapBeforeMeasure(), RemoveDiagonalGatesBeforeMeasure()]
    _choose_layout_0 = (
        TrivialLayout(coupling_map),
        Layout2qDistance(coupling_map, property_name="trivial_layout_score"),
    )
    _choose_layout_1 = (
        CSPLayout(coupling_map, call_limit=10000, time_limit=60)
    )
    _choose_layout_2 = SabreLayout(coupling_map, max_iterations=4, routing_pass=routing_pass)
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]
    _swap_check = CheckMap(coupling_map)
    _unroll = [
            UnitarySynthesis(
                G_QISKIT_GATE_SET,
#                approximation_degree=1,
                min_qubits=3,
            ),
            Unroll3qOrMore(),
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=G_QISKIT_GATE_SET),
            UnitarySynthesis(
                G_QISKIT_GATE_SET,
#                approximation_degree=1,
            ),
        ]
    # Define layout conditions
    def _choose_layout_condition(property_set):
        return not property_set["layout"]
    def _trivial_not_perfect(property_set):
        if property_set["trivial_layout_score"] is not None:
            if property_set["trivial_layout_score"] != 0:
                return True
        return False
    def _csp_not_found_match(property_set):
        if property_set["layout"] is None:
            return True
        if (
            property_set["CSPLayout_stop_reason"] is not None
            and property_set["CSPLayout_stop_reason"] != "solution found"
        ):
            return True
        return False
    # Create pass manager
    pm = PassManager()
    pm.append(_unroll3q)
    pm.append(_reset+_meas)
    if do_unroll:
        pm.append(_unroll)
    pm.append(_choose_layout_0, condition=_choose_layout_condition)
    pm.append(_choose_layout_1, condition=_trivial_not_perfect)
    pm.append(_choose_layout_2, condition=_csp_not_found_match)
    pm.append(_embed)
    pm.append(_swap_check)
    return pm

### BENCHMARK GENERATION -- Needed before executing compiler passes on a backend ###

def generate_benchmarks_for_backend(arch_file, backend_name, 
        mapped_circ_name=None, routing_pass=None, reset=False
):
    if mapped_circ_name is None:
        mapped_circ_name = 'base_mapping'
    # load coupling map
    coupling_map = read_arch_file(arch_file)
    # clear existing mapped circuits
    if not os.path.exists('%s/mapped_circuits' % BENCHMARK_PATH):
        os.mkdir('%s/mapped_circuits' % BENCHMARK_PATH)
    base_path = '%s/mapped_circuits/%s' % (BENCHMARK_PATH,backend_name)
    if reset:
        os.system('rm -rf %s' % base_path)
        os.mkdir('%s' % base_path)

    benchmark_files = [s for s in os.listdir('%s/base' % BENCHMARK_PATH)\
                            if s.endswith('.qasm') and 'meas' not in s]
    layout_pass_with_unroll = qiskitopt3_layout_pass(coupling_map)
    layout_pass_without_unroll = qiskitopt3_layout_pass(coupling_map, do_unroll=False)
    for qasm_file in benchmark_files:
        print(qasm_file)
        circ = QuantumCircuit.from_qasm_file('%s/base/%s' % (BENCHMARK_PATH,qasm_file))
        if circ.num_qubits > coupling_map.size():
            continue
        if 'cx' in circ.count_ops() and circ.count_ops()['cx'] > 10000:
            continue
        if 'qft' in qasm_file:
            mapped_circ = layout_pass_without_unroll.run(circ)
        else:
            mapped_circ = layout_pass_with_unroll.run(circ)
        if 'cx' in mapped_circ.count_ops() and mapped_circ.count_ops()['cx'] > 10000:
            continue
        qasm_name = qasm_file[:qasm_file.index('.qasm')]
        if not os.path.exists('%s/%s' % (base_path,qasm_file)):
            os.mkdir('%s/%s' % (base_path,qasm_file))
        mapped_qasm = mapped_circ.qasm()
        writer = open('%s/%s/%s.qasm' % (base_path,qasm_file,mapped_circ_name), 'w')
        writer.write(mapped_qasm)
        writer.close()
    print('done')

### SENSITIVITY ANALYSIS FOR FORESIGHT ###
# convergence_analysis: Can 100 iterations of Sabre reach the quality of ForeSight?
# insertion_analysis: How many steps does ForeSight need to complete execution?

def convergence_analysis(output_file):
    arch_file = '../arch/100grid.arch'
    backend = read_arch_file(arch_file)

    data = {}
    layout_pass = qiskitopt3_layout_pass(backend)
    for circ_name in ['bv_n50.qasm', 'bv_n100.qasm']:
        circ_path = '../benchmarks/base/%s' % circ_name
        circ = QuantumCircuit.from_qasm_file(circ_path)
        circ = layout_pass.run(circ)

        sabre = PassManager([
            TrivialLayout(backend),
            ApplyLayout(),
            SabreSwap(backend, heuristic='decay')
        ])

        foresight = PassManager([
            TrivialLayout(backend),
            ApplyLayout(),
            ForeSight(backend, slack=2, solution_cap=64, flags=FLAG_ALAP)
        ])

        best_sabre_circ = None
        best_foresight_circ = None
        d = {'sabre': [], 'foresight': []}
        for i in range(100):
            print('iteration %d' % i)
            sabre_circ = sabre.run(circ)
            sabre_circ = transpile(
                sabre_circ,
                basis_gates=G_QISKIT_GATE_SET,
                coupling_map=backend,
                layout_method='trivial',
                routing_method='none',
                optimization_level=0
            )
            if best_sabre_circ is None:
                best_sabre_cnots = np.inf
            else:
                best_sabre_cnots = best_sabre_circ.count_ops()['cx']
            sabre_cnots = sabre_circ.count_ops()['cx']
            if sabre_cnots < best_sabre_cnots:
                best_sabre_cnots = sabre_cnots
                best_sabre_circ = sabre_circ
            print('\tsabre: %d' % best_sabre_cnots)
            d['sabre'].append(best_sabre_cnots)

            foresight_circ = foresight.run(circ)
            foresight_circ = transpile(
                foresight_circ,
                basis_gates=G_QISKIT_GATE_SET,
                coupling_map=backend,
                layout_method='trivial',
                routing_method='none',
                optimization_level=0
            )
            if best_foresight_circ is None:
                best_foresight_cnots = np.inf
            else:
                best_foresight_cnots = best_foresight_circ.count_ops()['cx']
            foresight_cnots = foresight_circ.count_ops()['cx']
            if foresight_cnots < best_foresight_cnots:
                best_foresight_cnots = foresight_cnots
                best_foresight_circ = foresight_circ
            print('\tforesight: %d' % best_foresight_cnots)
            d['foresight'].append(best_foresight_cnots)
        data[circ_name] = d
    writer = open(output_file, 'wb')
    pickle.dump(data, writer)
    writer.close()

def insertion_analysis(output_file):
    arch_file = '../arch/ibm_tokyo.arch'
    backend = read_arch_file(arch_file)

    circ_file = '../benchmarks/base/vqe_n8.qasm'
    circ = QuantumCircuit.from_qasm_file(circ_file)

    layout_pass = qiskitopt3_layout_pass(backend)
    circ = layout_pass.run(circ)

    data = {}
    sabre_compiler = StatSABRE(backend, heuristic='decay')
    foresight_compiler = ForeSight(backend, slack=2, solution_cap=64, flags=FLAG_ALAP)

    PassManager([TrivialLayout(backend), ApplyLayout(), sabre_compiler]).run(circ)
    PassManager([TrivialLayout(backend), ApplyLayout(), foresight_compiler]).run(circ)
    data['sabre'] = sabre_compiler.swap_segments
    data['foresight'] = foresight_compiler.swap_segments

    print(sabre_compiler.swap_segments)
    print(sum(sabre_compiler.swap_segments))
    print(foresight_compiler.swap_segments)
    print(sum(foresight_compiler.swap_segments))

    writer = open(output_file, 'wb')
    pickle.dump(data, writer)
    writer.close()

### SENSITIVITY ANALYSIS ###
# tmsens: Time and Memory analysis on BV-50, BV-100 circuits. Sweep on ForeSight parameters.
# bvsens: Time and Memory analysis for scalability.
# gensens: General overhead sensitivity analysis. Sweep on ForeSight parameters. Google Sycamore.

TMSENS = [
    'bv_n50.qasm',
    'bv_n100.qasm'
]

BVSENS = [
    'bv_n100.qasm',
    'bv_n200.qasm',
    'bv_n500.qasm'
]

GENSENS = [
    'cycle10_2_110.qasm',
    'adr4_197.qasm',
    'hwb4_49.qasm',
    'sym9_148.qasm',
    '4_49_16.qasm',
    'wim_266.qasm',
    'square_root_7.qasm',
    'hwb4_49.qasm',
    'life_238.qasm'
]

NOISEBENCH = [
    'bv_n14.qasm',
    'adder_n10.qasm',
    'seca_n11.qasm',
    'qf21_n15.qasm',
    'ba_20_1_1.qasm',
    'multiply_n13.qasm',
    'multiplier_n15.qasm',
    'qpe_n9.qasm',
    'hwb4_49_meas.qasm',
    'wim_266_meas.qasm'
]

def generate_sens_benchmarks(sens_folder, circuits, arch_file,
        base_folder='base', mapped_circ_name=None, routing_pass=None, reset=False
):
    if mapped_circ_name is None:
        mapped_circ_name = 'base_mapping'
    coupling_map = read_arch_file(arch_file)

    base_path = '%s/%s' % (BENCHMARK_PATH, sens_folder)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if reset:
        os.system('rm -rf %s' % base_path)
        os.mkdir('%s' % base_path)
    layout_pass = qiskitopt3_layout_pass(coupling_map)

    if circuits is None:
        circuits = [f for f in os.listdir(os.path.join(BENCHMARK_PATH, base_folder)) 
                    if f.endswith('.qasm')]

    for qasm_file in circuits:
        print(qasm_file)
        circ = QuantumCircuit.from_qasm_file('%s/%s/%s' % (BENCHMARK_PATH, base_folder, qasm_file)) 
        mapped_circ = layout_pass.run(circ)
        if not os.path.exists('%s/%s' % (base_path,qasm_file)):
            os.mkdir('%s/%s' % (base_path,qasm_file))
            mapped_qasm = mapped_circ.qasm()
            writer = open('%s/%s/%s.qasm' % (base_path,qasm_file,mapped_circ_name), 'w')
            writer.write(mapped_qasm)
            writer.close()

### BENCHMARK EXECUTION ###

def benchmark_circuits(folder, arch_file, router_name, routing_func, runs=5, memory=True):
    backend = read_arch_file(arch_file)
    benchmark_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    for subfolder in benchmark_folders:
        print(subfolder)
        bench_path = os.path.join(folder, subfolder)
        # Input qasm file
        file_path = os.path.join(bench_path, 'base_mapping.qasm')
        # Output qasm file
        output_path = os.path.join(bench_path, '%s_circ.qasm' % router_name)
        # Cnot difference, Time, and Memory file
        data_path = os.path.join(bench_path, '%s_data.txt' % router_name)
        if os.path.exists(data_path):
            print('Skipping %s' % subfolder)
            continue
        # Declare data structures
        time_array = []
        memory_array = []
        best_circ = None 
        # Get input circuit
        base_circ = QuantumCircuit.from_qasm_file(file_path)
        for r in range(runs):
            circ = base_circ.copy()
            # Benchmark here
            if memory:
                tracemalloc.start(1)
                routing_func(circ,arch_file)  # Do not save output.
                _, peak_mem = tracemalloc.get_traced_memory()
                peak_mem -= tracemalloc.get_tracemalloc_memory()
                tracemalloc.stop()
            start = time.time()
            output_circ = routing_func(circ, arch_file)
            end = time.time()
            # Transpile circuit with O0 to basis gate set.
            output_circ = transpile(
                output_circ,
                coupling_map=backend,
                basis_gates=G_QISKIT_GATE_SET,
                layout_method='trivial',
                routing_method='none',
                optimization_level=0
            )
            if 'cx' not in output_circ.count_ops():
                cnots = 0
            else:
                cnots = output_circ.count_ops()['cx']
            depth = output_circ.depth()
            if best_circ is None:
                best_circ = output_circ
            else:
                if 'cx' not in best_circ.count_ops():
                    min_cnots = 0
                else:
                    min_cnots = best_circ.count_ops()['cx']
                min_depth = best_circ.depth()
                if cmp(cnots, min_cnots, depth, min_depth):
                    best_circ = output_circ
            # Record data
            time_array.append((end - start) * 1000)  # (end-start) is in seconds -- want in ms
            if memory:
                memory_array.append(peak_mem)  # peak_mem is in bytes
        # Print out data to stdout
        if 'cx' not in best_circ.count_ops():
            best_cnots = 0
        else:
            best_cnots = best_circ.count_ops()['cx']
        if 'cx' not in base_circ.count_ops():
            base_cnots = 0
        else:
            base_cnots = base_circ.count_ops()['cx']
        print('\tcnot difference: %d' % (best_cnots - base_cnots))
        print('\tdepth: %d' % best_circ.depth())
        print('\tmean time: %.3f' % np.mean(time_array))
        print('\tmean memory: %.3f' % np.mean(memory_array))
        # Record best circ in qasm file
        writer = open(output_path, 'w')
        writer.write(best_circ.qasm())
        writer.close()
        # Record data in data path
        writer = open(data_path, 'w')
        writer.write('cnots\t%d\n' % best_cnots)
        writer.write('depth\t%d\n' % best_circ.depth())
        writer.write('time\t%.3f\n' % np.mean(time_array))
        writer.write('memory\t%.3f\n' % np.mean(memory_array))
        writer.close()

### DATA COMPILATION ###
# Compiles all circuits using O3 after routing.
# Data is saved within a pickle and a CSV.

BASE_COMPILERS=['sabre','foresight','astar']
def compile_data(folder, arch_file, csv_file, pickle_file, compilers=BASE_COMPILERS, 
        use_O3=True, circuits=None):
    if circuits is None:
        benchmark_folder = [d for d in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, d))]
    else:
        benchmark_folder = circuits
    csv_data = defaultdict(list)
    pkl_data = {}
    base_columns = ['cnots added', 'depth', 'time', 'memory', 'cnots added O3', 'depth O3']
    columns = []
    for cat in compilers:
        for b in base_columns:
            columns.append('%s %s' % (cat, b))
    columns.insert(0, 'original cnots')
    backend = read_arch_file(arch_file)
    for (i,subfolder) in enumerate(benchmark_folder):
        print(subfolder)
        bench_path = os.path.join(folder, subfolder)
        base_circ = QuantumCircuit.from_qasm_file(os.path.join(bench_path, 'base_mapping.qasm'))
        if 'cx' not in base_circ.count_ops():
            base_cnots = 0
        else:
            base_cnots = base_circ.count_ops()['cx']
        base_depth = base_circ.depth()
        csv_data['original cnots'].append(base_cnots)
        pkl_data[subfolder] = {
            'original': {
                'gate count': base_circ.size(),
                'original cnots': base_cnots,
                'original depth': base_depth,
                'original qasm': base_circ.qasm()
            }
        }
        for cat in compilers:
            pkl_data[subfolder][cat] = {}
            print('\treading %s' % cat)
            data_file = os.path.join(bench_path, '%s_data.txt' % cat)
            reader = open(data_file, 'r')
            line = reader.readline()
            while line != '':
                print('\t\t%s' % line.strip())
                line_data = line.split('\t')
                dtype = line_data[0]
                if dtype == 'cnots' or dtype == 'depth':
                    value = int(line_data[1])
                else:
                    value = float(line_data[1])
                if dtype == 'cnots':
                    csv_data['%s cnots added' % cat].append(value - base_cnots)
                    pkl_data[subfolder][cat]['cnots added'] = value - base_cnots
                else:
                    csv_data['%s %s' % (cat, dtype)].append(value)
                    if dtype == 'depth':
                        pkl_data[subfolder][cat]['final depth'] = value
                    elif dtype == 'time':
                        pkl_data[subfolder][cat]['execution time'] = value
                    elif dtype == 'memory':
                        pkl_data[subfolder][cat]['memory'] = value
                line = reader.readline()
            reader.close()
            print('\tperforming O3 on %s' % cat)
            original_circ = QuantumCircuit.from_qasm_file(
                        os.path.join(bench_path, '%s_circ.qasm' % cat))
            try:
                if use_O3:
                    trans_circ = transpile(
                        original_circ,
                        basis_gates=G_QISKIT_GATE_SET,
                        coupling_map=backend,
                        layout_method='trivial',
                        routing_method='none',
                        optimization_level=3
                    )
                else:
                    trans_circ = original_circ
            except:
                trans_circ = original_circ
            if trans_circ.size() == 0:
                tc_cnots = np.inf  # invalid circuit
            elif 'cx' not in trans_circ.count_ops():
                tc_cnots = 0
            else:
                tc_cnots = trans_circ.count_ops()['cx']
            tc_depth = trans_circ.depth()
            csv_data['%s cnots added O3' % cat].append(tc_cnots-base_cnots)
            csv_data['%s depth O3' % cat].append(tc_depth)
            pkl_data[subfolder][cat]['cnots added O3'] = tc_cnots-base_cnots
            pkl_data[subfolder][cat]['final depth O3'] = tc_depth
            pkl_data[subfolder][cat]['mapped qasm'] = original_circ.qasm()
            pkl_data[subfolder][cat]['final qasm'] = trans_circ.qasm()
            if tc_cnots == np.inf:
                print('\tinvalid circuit')
            else:
                print('\ttranspiled cnots = %d, depth = %d' % (tc_cnots-base_cnots, tc_depth))
    df = pd.DataFrame(data=csv_data, index=benchmark_folder)
    df.to_csv(csv_file)

    writer = open(pickle_file, 'wb')
    pickle.dump(pkl_data, writer)
    writer.close()

def merge_pickles(output_file, *input_files):
    all_data = {}
    dict_array = []
    for f in input_files:
        reader = open(f, 'rb')
        d = pickle.load(reader)
        reader.close()
        dict_array.append(d)
    all_keys = set(dict_array[0].keys()).intersection(*[set(d.keys()) for d in dict_array[1:]])
    for i in range(len(dict_array)):
        f = input_files[i]
        src = f[0:f.index('.pkl')] 
        d = dict_array[i]
        filtered_dict = {}
        for k in all_keys:
            filtered_dict[k] = d[k]
        all_data[src] = filtered_dict
    writer = open(output_file, 'wb')
    pickle.dump(all_data, writer)
    writer.close()

def merge_noise_sim_pickles(input_folder, output_file):
    base_path = os.path.join(BENCHMARK_PATH, input_folder)
    benchmarks = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,d))]

    data = {}
    for subfolder in benchmarks:
        print(subfolder)
        counts_file = '%s/%s/counts.pkl' % (base_path, subfolder)
        if not os.path.exists(counts_file):
            continue
        try:
            reader = open(counts_file, 'rb')
            counts = pickle.load(reader)
            reader.close()
        except:
            continue
        # Read the qasm files
        for cat in ['sabre', 'foresight_alap', 'noisy_foresight_alap',\
                    'foresight_asap', 'noisy_foresight_asap']:
            circ = QuantumCircuit.from_qasm_file(
                        '%s/%s/%s_circ.qasm' % (base_path, subfolder, cat))
            if 'cx' not in circ.count_ops():
                cnots = 0
            else:
                cnots = circ.count_ops()['cx']
            print('\t%s fidelity: %f' % (cat, counts[cat]['fidelity']))
            print('\t%s ist: %f' % (cat, counts[cat]['ist']))
            print('\t%s cnots: %d' % (cat, cnots))
            counts[cat]['cnots'] = cnots

        data[subfolder] = counts
    writer = open(output_file, 'wb')
    pickle.dump(data, writer)
    writer.close()

### Routing Execution Functions. See fs_batchrun for usage. ###

def _sabre_route(circ, arch_file):
    backend = read_arch_file(arch_file)
    compiler = SabreSwap(backend, heuristic='decay')
    sabre_pass = PassManager([
        TrivialLayout(backend),
        ApplyLayout(),
        compiler
    ])
    return sabre_pass.run(circ)

FORESIGHT_DEFAULT_FLAGS = FLAG_ALAP
def _foresight_route(circ, arch_file, compiler):
    backend = read_arch_file(arch_file)
    foresight_pass = PassManager([
        TrivialLayout(backend),
        ApplyLayout(),
        compiler
    ])
    return foresight_pass.run(circ)

def _astar_route(circ, arch_file):
    circ = RemoveBarriers()(circ)
    circ = RemoveFinalMeasurements()(circ)
    output_circ = QuantumCircuit(1,1)
    try:
        results = mqt.qmap.compile(
                circ, arch_file, method='heuristic', initial_layout='identity')
        results = results.json()
        output_circ = QuantumCircuit.from_qasm_str(results['mapped_circuit']['qasm'])
    except Exception as e:
        print(e)
    return output_circ

def _lookahead_route(circ, arch_file):
    backend = read_arch_file(arch_file)
    compiler = LookaheadSwap(backend)
    lookahead_pass = PassManager([
        TrivialLayout(backend),
        ApplyLayout(),
        compiler
    ])
    return lookahead_pass.run(circ)

from pytket import Circuit as TketCircuit
from pytket import OpType
from pytket.circuit import Qubit as TketQubit
from pytket.circuit import Node as TketNode
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
try:  # Need to do this because PACE has a lower python version.
    from pytket.architecture import Architecture
    from pytket.placement import Placement
except ImportError:
    from pytket.routing import Architecture
    from pytket.routing import Placement
from pytket.transform import CXConfigType
import pytket.passes

def _tket_route(circ, arch_file):
    qasm = circ.qasm()
    tket_circ = circuit_from_qasm_str(qasm)
    backend = Architecture(read_arch_file(arch_file).get_edges())
    # Perform Peephole Optimization
    tket_basis_1q = {OpType.Rx, OpType.Ry, OpType.Rz}
    tket_basis_2q = {OpType.CX}
    trivial_mapping = {TketQubit(i):TketNode(i) for i in range(tket_circ.n_qubits)}
    # Create basis synthesis circuits
    cx_to_cx = TketCircuit(2)
    cx_to_cx.CX(0,1)
    def tk1_to_sq(x,y,z):
        tmp = TketCircuit(1)
        tmp.Rx(x,0).Ry(y,0).Rz(z,0)
        return tmp
    # Perform other optimizations.
    pytket.passes.DecomposeBoxes().apply(tket_circ)
    pytket.passes.FullPeepholeOptimise().apply(tket_circ)
    #pytket.passes.RebaseCustom(tket_basis_2q, cx_to_cx, tk1_to_sq).apply(tket_circ)
    placement = Placement(backend)
    placement.place_with_map(tket_circ, trivial_mapping)
    pytket.passes.RoutingPass(backend).apply(tket_circ)
    try:
        pytket.passes.DelayMeasures().apply(tket_circ)
        pytket.passes.CXMappingPass(backend, placement).apply(tket_circ)
        return QuantumCircuit.from_qasm_str(circuit_to_qasm_str(tket_circ))
    except:
        return QuantumCircuit(1,1)

def _z3_route(circ, arch_file):
    circ = RemoveBarriers()(circ)
    circ = RemoveFinalMeasurements()(circ)
    output_circ = QuantumCircuit(1,1)
    try:
        results = mqt.qmap.compile(
                circ, arch_file, method='exact', initial_layout='identity',
                encoding='bimander', commander_grouping='fixed3')
        results = results.json()
        output_circ = QuantumCircuit.from_qasm_str(results['mapped_circuit']['qasm'])
    except Exception as e:
        print(e)
    return output_circ

def _bip_route(circ, arch_file):
    backend = read_arch_file(arch_file)
    compiler = BIPMapping(backend)
    bip_pass = PassManager([
        TrivialLayout(backend),
        ApplyLayout(),
        compiler
    ])
    return bip_pass.run(circ)

from olsq import OLSQ
from olsq.device import qcdevice

def _olsq_route(circ, arch_file):
    backend = read_arch_file(arch_file)

    solver = OLSQ('swap', 'transition')
    solver.setdevice(qcdevice(name='dev', nqubits=backend.size(),\
                        connection=[tuple(e) for e in backend.get_edges()], swap_duration=3))
    solver.setprogram(circ.qasm())
    res, _, _ = solver.solve()
    return QuantumCircuit.from_qasm_str(res)

### MORE DATA COMPILATION ###

DATA_FOLDER_PATH = '../data/'
DATA_BENCH_PATH = '%s/benchmarks' % DATA_FOLDER_PATH
DATA_SENS_PATH = '%s/sensitivity' % DATA_FOLDER_PATH

def compile_all_benchmarks(compilers=['sabre','foresight_alap','foresight_asap','tket','astar']):
    compile_data('../benchmarks/mapped_circuits/ibm_tokyo',
            '../arch/ibm_tokyo.arch', '../data/benchmarks/csv/ibm_tokyo.csv',
            'ibm_tokyo.pkl', compilers=compilers, use_O3=True)
    compile_data('../benchmarks/mapped_circuits/google_sycamore',
            '../arch/google_sycamore.arch', '../data/benchmarks/csv/google_sycamore.csv',
            'google_sycamore.pkl', compilers=compilers, use_O3=True)
    compile_data('../benchmarks/mapped_circuits/rigetti_aspen9',
            '../arch/rigetti_aspen9.arch', '../data/benchmarks/csv/rigetti_aspen9.csv',
            'rigetti_aspen9.pkl', compilers=compilers, use_O3=True)
    compile_data('../benchmarks/mapped_circuits/ibm_toronto',
            '../arch/ibm_toronto.arch', '../data/benchmarks/csv/ibm_toronto.csv',
            'ibm_toronto.pkl', compilers=compilers, use_O3=True)
    compile_data('../benchmarks/mapped_circuits/ibm_heavyhex',
            '../arch/ibm_3heavyhex.arch', '../data/benchmarks/csv/ibm_heavyhex.csv',
            'ibm_heavyhex.pkl', compilers=compilers, use_O3=True)
    merge_pickles('benchmarks.pkl', 'ibm_tokyo.pkl', 'google_sycamore.pkl', 'rigetti_aspen9.pkl',\
                    'ibm_toronto.pkl', 'ibm_heavyhex.pkl')
    os.system('mv *.pkl ../data/benchmarks')

def compile_all_sensitivity(tmsens=True, bvsens=True, gensens=True, ins=True, conv=True,\
        solver=True
):
    if tmsens:
        tmsens_input_path = '%s/sensitivity/tmsens_100grid' % BENCHMARK_PATH
        tmsens_csv_path = '%s/tmsens/csv/tmsens.csv' % DATA_SENS_PATH
        tmsens_pkl_path = '%s/tmsens/tmsens.pkl' % DATA_SENS_PATH
        tmsens_compilers = []
        for d in [0,1,2,3,4]:
            for s in [4,8,16,32,64,128]:
                tmsens_compilers.append('fs_%d_%d' % (d,s))
        compile_data(tmsens_input_path, '../arch/google_weber.arch',
                tmsens_csv_path, tmsens_pkl_path, compilers=tmsens_compilers, use_O3=False)
    if bvsens:    
        bvsens_input_path = '%s/sensitivity/bvsens_500grid' % BENCHMARK_PATH
        bvsens_csv_path = '%s/bvsens/csv/bvsens.csv' % DATA_SENS_PATH
        bvsens_pkl_path = '%s/bvsens/bvsens.pkl' % DATA_SENS_PATH
        compile_data(bvsens_input_path, '../arch/500grid.arch', 
                bvsens_csv_path, bvsens_pkl_path, compilers=['sabre','foresight'], use_O3=False)
    if gensens:
        gensens_input_path = '%s/sensitivity/gensens_sycamore' % BENCHMARK_PATH
        gensens_csv_path = '%s/gensens/csv/gensens.csv' % DATA_SENS_PATH
        gensens_pkl_path = '%s/gensens/gensens.pkl' % DATA_SENS_PATH
        gensens_compilers = []
        for s in [4,8,16,32,64,128]:
            gensens_compilers.append('fs_%d_%d' % (2,s))
        gensens_compilers.append('sabre')
        compile_data(gensens_input_path, '../arch/google_weber.arch',
                gensens_csv_path, gensens_pkl_path, compilers=gensens_compilers, use_O3=False)
    if ins:
        insertion_analysis('%s/foresight_vs_sabre_swpins.pkl' % DATA_SENS_PATH)
    if conv:
        convergence_analysis('%s/foresight_vs_sabre_100iter.pkl' % DATA_SENS_PATH)
    if solver:
        solver_input_path = '%s/solver_circuits/ibm_manila' % BENCHMARK_PATH
        solver_csv_path = '%s/solver/csv/solver.csv' % DATA_SENS_PATH
        solver_pkl_path = '%s/solver/solver.pkl' % DATA_SENS_PATH
        solver_compilers = ['foresight_alap','foresight_asap','sabre',\
                            'z3solver','bipsolver','olsq','lookahead']
        compile_data(solver_input_path, '../arch/ibm_manila.arch',\
                solver_csv_path, solver_pkl_path, compilers=solver_compilers, use_O3=False)

### TOQM COMPARISION ###

TOQM_CIRCUITS = [
    'cycle10_2_110.qasm',
    'adr4_197.qasm',
    'hwb4_49.qasm',
    'vqe_n8.qasm',
    'sym9_148.qasm',
    '4_49_16.qasm',
    'wim_266.qasm',
    'square_root_7.qasm',
]

def compile_for_analytical_model(output_file):
    base_path = '%s/mapped_circuits/google_sycamore' % BENCHMARK_PATH
    backend = read_arch_file('../arch/google_sycamore.arch')
    data = {}
    for circ_name in TOQM_CIRCUITS:
        base_circ = QuantumCircuit.from_qasm_file(
                '%s/%s/base_mapping.qsm' % (base_path, circ_name))
        data[circ_name] = {
            'qasm': base_circ.qasm(),
            'base cnots': base_circ.count_ops()['cx'],
            'base depth': base_circ.depth()
        }
        for policy in ['sabre','foresight','astar','tket','toqm']:
            circ = QuantumCircuit.from_qasm_file(
                    '%s/%s/%s_circ.qasm' % (base_path, circ_name, policy))
            circ = transpile(circ,
                    basis_gates=G_QISKIT_GATE_SET,
                    coupling_map=backend,
                    layout_method='trivial',
                    routing_method='none',
                    optimization_level=3)
            data[circ_name][policy] = {
                'qasm': circ.qasm(),
                'cnots added': circ.count_ops()['cx'],
                'final depth': circ.depth()
            }
    writer = open(output_file, 'wb')
    pickle.dump(data, writer)
    writer.close()

def compile_toqm():
    arch_names = ['google_sycamore']

    for arch_name in arch_names:
        toqm_input_path = '%s/mapped_circuits/%s' % (BENCHMARK_PATH, arch_name)
        toqm_csv_path = '%s/toqm/csv/toqm_%s.csv' % (DATA_BENCH_PATH, arch_name)
        toqm_pkl_path = '%s/toqm/toqm_%s.pkl' % (DATA_BENCH_PATH, arch_name)

        compile_data(toqm_input_path, '../arch/%s.arch' % arch_name,
                toqm_csv_path, toqm_pkl_path, compilers=['sabre', 'foresight', 'toqm'], 
                use_O3=True, circuits=TOQM_CIRCUITS)

def compare_eps_general(arch_name, compilers, circuits,
        coh_t1=15000, cxtime=32, sqtime=25, cxerror=0.01, sqerror=0.001, roerror=0.02, use_O3=True):
    base_path = '%s/mapped_circuits/%s' % (BENCHMARK_PATH,arch_name)
    # Compute EPS for each circuit
    backend = read_arch_file('../arch/%s.arch' % arch_name)
    for circ_name in circuits:
        base_circ = QuantumCircuit.from_qasm_file(
                    '%s/%s/base_mapping.qasm' % (base_path, circ_name))
        base_cnots = base_circ.count_ops()['cx']
        print(circ_name)
        for policy in compilers:
            circ = QuantumCircuit.from_qasm_file(
                    '%s/%s/%s_circ.qasm' % (base_path, circ_name, policy))
            circ = transpile(circ,
                    basis_gates=G_QISKIT_GATE_SET,
                    coupling_map=backend,
                    layout_method='trivial',
                    routing_method='none',
                    optimization_level=3 if use_O3 else 0)
            eps = 1.0
            logeps = 0.0
            num_cnots = circ.count_ops()['cx']
            eps *= (1.0-cxerror)**num_cnots  # may underflow, logeps is better value
            logeps += -np.log(1.0-cxerror)*num_cnots
            # We can ignore single qubit and readout error: they will cancel out.
            time = 0.0
            # Compute circuit time
            dag = circuit_to_dag(circ)
            active = set()
            for layer in dag.layers():
                layer_data = layer['graph']
                max_time = 0
                for node in layer_data.front_layer():
                    if len(node.qargs) == 2:
                        max_time = cxtime
                    elif len(node.qargs) == 1 and node.name != 'rz':
                        max_time = sqtime
                time += max_time
                for lst in layer['partition']:
                    for q in lst:
                        active.add(q)
            eps *= np.exp(-time*len(active)/coh_t1)
            logeps += time/coh_t1 * len(active)
            print('\tcnots and depth for %s on %s: %d, %d' 
                    % (policy, circ_name, circ.count_ops()['cx'], circ.depth()))
            print('\t\tcx overhead: %d' % (circ.count_ops()['cx']-base_cnots))
            print('\t\tlatency: %d' % time)
            print('\t\tqubits used: %d' % len(active))
            print('\t\tEPS: %.3e (%.3e)' % (eps, logeps))

### BMT COMPARISON ###

BMT_CIRCUITS = [
    'hwb4_49.qasm',
    'wim_266.qasm',
    'f2_232.qasm',
    'misex1_241.qasm',
    'radd_250.qasm',
    'cycle10_2_110.qasm',
    'square_root_7.qasm',
    'vqe_n8.qasm',
    'sym9_148.qasm',
]

def compare_foresight_bmt():
    backend = read_arch_file('../arch/ibm_tokyo.arch')
    for circ_name in BMT_CIRCUITS:
        print('[%s]' % circ_name)
        base_circ = QuantumCircuit.from_qasm_file('%s/mapped_circuits/ibm_tokyo/%s/base_mapping.qasm'\
                            % (BENCHMARK_PATH, circ_name))
        base_cnots = base_circ.count_ops()['cx']
        for cat in ['foresight_alap', 'bmt']:
            circ = QuantumCircuit.from_qasm_file('%s/mapped_circuits/ibm_tokyo/%s/%s_circ.qasm'\
                            % (BENCHMARK_PATH, circ_name, cat))
            circ = transpile(circ,
                    basis_gates=G_QISKIT_GATE_SET,
                    optimization_level=3)
            cnots = circ.count_ops()['cx']
            depth = circ.depth()
            print('\t%s: cnot overhead=%d, depth=%d' % (cat, cnots-base_cnots, depth))

def compute_cnot_dist_stats(arch_name, arch_file):
    reader = open('../data/benchmarks/benchmarks.pkl', 'rb')
    benchmarks = pickle.load(reader)
    reader.close()

    coupling_map = read_arch_file(arch_file)

    circ_dists = []
    for circ_name in benchmarks[arch_name]:
        circ = QuantumCircuit.from_qasm_str(\
                benchmarks[arch_name][circ_name]['original']['original qasm'])
        dist = 0
        num_cx = 0
        for (ins, qargs, _) in circ.data:
            if ins.name == 'cx':
                q0,q1 = qargs
                dist += coupling_map.distance_matrix[q0.index, q1.index]
                num_cx += 1
        if num_cx > 0:
            dist /= num_cx
            circ_dists.append(dist)
        print('\t[%s] dist = %f' % (circ_name, dist))
    print('dist mean: %f, stdev: %f' % (np.mean(circ_dists), np.std(circ_dists)))

### IBMQ LOOKAHEAD ###

LA_CIRCUITS = [
    '4_49_16.qasm',
    'hwb4_49.qasm',
    'square_root_7.qasm',
    'vqe_n8.qasm',
    'life_238.qasm'
]

def benchmark_ibmq_la(folder, arch_file):
    for circ_name in LA_CIRCUITS:
        circ = QuantumCircuit.from_qasm_file('%s/%s/base_mapping.qasm' % (folder, circ_name))
        start = time.time()
        lookahead_circ = _lookahead_route(circ, arch_file)
        end = time.time()
        print('time taken for %s: %f' % (circ_name, (end-start)*1000))
        writer = open('%s/%s/la_circ.qasm' % (folder, circ_name), 'w')
        writer.write(lookahead_circ.qasm())
        writer.close()

def ibmq_la_try_hwb4_49(arch_file):
    coupling_map = read_arch_file('../arch/ibmq_guadalupe.arch')
    circ = QuantumCircuit.from_qasm_file('%s/base/hwb4_49.qasm' % BENCHMARK_PATH)
    start = time.time()
    t_circ = transpile(circ, coupling_map=coupling_map, optimization_level=3,\
                        routing_method='lookahead')
    end = time.time()
    cnots_added = t_circ.count_ops()['cx'] - circ.count_ops()['cx']
    print('Time taken: %f' % ((end-start)*1000))
    print('CNOTS added: %d' % cnots_added)

### OTHER ###

def compute_cnot_dist_all_archs():
    for arch_name in ['ibm_tokyo', 'ibm_toronto', 'google_sycamore', 'rigetti_aspen9']:
        arch_file = '../arch/%s.arch' % arch_name
        print(arch_name)
        compute_cnot_dist_stats(arch_name, arch_file)

def prob_of_success(pgate,Ng,n,depth,T1_time,pmeas=0.01,gate_duration=100):
	'''
	pgate-> gate error prob
	pmeas-> measurement error prob	[Default 1%]
	Ng   -> Number of CNOT gates	
	n	 -> Number of Qubits in program
	depth-> Number of gates in the critical path
	gate duration -> 100 ns
	T1_time -> T1 time in microseconds
	'''

	prob_no_gate_error = (1-pgate)**Ng
	prob_no_meas_error = (1-pmeas)**n
	prob_no_coherence_error = np.exp(-depth*gate_duration/(T1_time*1000))
	prob_success = prob_no_gate_error * prob_no_meas_error * prob_no_coherence_error

	return prob_success

def compute_psuccess_ratio(cnot_error_rate_toqm,cnot_error_rate_fs,
        Ng_toqm,Ng_fs,n,depth_toqm,depth_fs,T1,pmeas=0.01,gate_duration=100):
	'''
	cnot_error_rate -> gate error rate for cnot operations
	Ng_toqm			-> Number of CNOTs in TOQM
	Ng_fs			-> Number of CNOTs in ForeSight
	n				-> Number of program qubits
	depth_toqm 		-> Number of gates in the critical path for TOQM code
	depth_fs 		-> Number of gates in the critical path for FS code
	T1				-> Device T1 time
	'''
	psuccess_toqm = prob_of_success(pgate=cnot_error_rate_toqm,
            Ng=Ng_toqm,n=n,depth=depth_toqm,T1_time=T1,pmeas=pmeas,gate_duration=gate_duration)
	psuccess_fs = prob_of_success(pgate=cnot_error_rate_fs,
            Ng=Ng_fs,n=n,depth=depth_fs,T1_time=T1,pmeas=pmeas,gate_duration=gate_duration)
	
	return psuccess_fs/psuccess_toqm
