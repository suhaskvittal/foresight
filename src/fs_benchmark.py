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
                            if s.endswith('.qasm')]
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

TMSENS = [
    'vqe_n8.qasm'
]

BVSENS = [
    'bv_n100.qasm',
    'bv_n200.qasm',
    'bv_n500.qasm'
]

GENSENS = [
    'cm85a_209.qasm',
    'sqn_258.qasm',
    'misex1_241.qasm',
    'hwb5_53.qasm',
    'rd53_251.qasm',
    'f2_232.qasm',
    'sf_276.qasm',
    'sym9_146.qasm',
    'mini-alu_167.qasm',
    'mod10_171.qasm',
]

NOISEBENCH = [
    'bv_n14.qasm',
    'adder_n10.qasm',
    'seca_n11.qasm',
    'qf21_n15.qasm',
    'ba_20_1_1.qasm',
    'multiply_n13.qasm',
    'multiplier_n15.qasm',
    'qpe_n9.qasm'
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

def benchmark_circuits(folder, arch_file, router_name, routing_func, runs=5, memory=False):
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

BASE_COMPILERS=['sabre','foresight','astar']
def compile_data(folder, arch_file, csv_file, pickle_file, compilers=BASE_COMPILERS, use_O3=True):
    benchmark_folder = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
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
            print('\transpiled cnots = %d, depth = %d' % (tc_cnots-base_cnots, tc_depth))
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

def merge_noise_sim_pickles(output_file):
    base_path = os.path.join(BENCHMARK_PATH, 'noise')
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
        for cat in ['sabre', 'foresight', 'noisy_foresight']:
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

from pytket import Circuit as TketCircuit
from pytket import OpType
from pytket.circuit import Qubit as TketQubit
from pytket.circuit import Node as TketNode
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
from pytket.architecture import Architecture
from pytket.placement import Placement
#from pytket.routing import Architecture
#from pytket.routing import Placement
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
