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
    
def qiskitopt3_layout_pass(coupling_map, routing_pass=None):
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
    layout_pass = qiskitopt3_layout_pass(coupling_map)
    for qasm_file in benchmark_files:
        print(qasm_file)
        circ = QuantumCircuit.from_qasm_file('%s/base/%s' % (BENCHMARK_PATH,qasm_file))
        if circ.num_qubits > coupling_map.size():
            continue
        if 'cx' in circ.count_ops() and circ.count_ops()['cx'] > 10000:
            continue
        mapped_circ = layout_pass.run(circ)
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

BVSENS1 = [
    'bv_n50.qasm'
]

BVSENS2 = [
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
        mapped_circ_name=None, routing_pass=None, reset=False
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
    for qasm_file in circuits:
        print(qasm_file)
        circ = QuantumCircuit.from_qasm_file('%s/base/%s' % (BENCHMARK_PATH, qasm_file)) 
        mapped_circ = layout_pass.run(circ)
        if not os.path.exists('%s/%s' % (base_path,qasm_file)):
            os.mkdir('%s/%s' % (base_path,qasm_file))
        mapped_qasm = mapped_circ.qasm()
        writer = open('%s/%s/%s.qasm' % (base_path,qasm_file,mapped_circ_name), 'w')
        writer.write(mapped_qasm)
        writer.close()

def benchmark_circuits(folder, arch_file, router_name, routing_func, runs=5):
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
        tracemalloc.start(4)
        for r in range(runs):
            circ = base_circ.copy()
            # Benchmark here
            tracemalloc.reset_peak()
            start = time.time()
            output_circ = routing_func(circ, arch_file)
            end = time.time()
            _, peak_mem = tracemalloc.get_traced_memory()
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
            memory_array.append(peak_mem)  # peak_mem is in bytes
        tracemalloc.stop()
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

def build_csv_file(folder, arch_file, output_file):
    benchmark_folder = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    data = defaultdict(list)
    base_columns = ['cnots added', 'depth', 'time', 'memory', 'cnots added O3', 'depth O3']
    columns = []
    compilers = ['sabre', 'fsalap', 'astar']
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
        data['original cnots'].append(base_cnots)
        for cat in compilers:
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
                    data['%s cnots added' % cat].append(value - base_cnots)
                else:
                    data['%s %s' % (cat, dtype)].append(value)
                line = reader.readline()
            reader.close()
            print('\tperforming O3 on %s' % cat)
            original_circ = QuantumCircuit.from_qasm_file(
                        os.path.join(bench_path, '%s_circ.qasm' % cat))
            try:
                trans_circ = transpile(
                    original_circ,
                    basis_gates=G_QISKIT_GATE_SET,
                    coupling_map=backend,
                    layout_method='trivial',
                    routing_method='none',
                    optimization_level=3
                )
            except:
                trans_circ = original_circ
            if 'cx' not in trans_circ.count_ops():
                tc_cnots = 0
            else:
                tc_cnots = trans_circ.count_ops()['cx']
            tc_depth = trans_circ.depth()
            data['%s cnots added O3' % cat].append(tc_cnots-base_cnots)
            data['%s depth O3' % cat].append(tc_depth)
            print('\transpiled cnots = %d, depth = %d' % (tc_cnots-base_cnots, tc_depth))
    df = pd.DataFrame(data=data, index=benchmark_folder)
    df.to_csv(output_file)

def noise_simulation(folder):
    pass

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
    except:
        pass
    return QuantumCircuit.from_qasm_str(circuit_to_qasm_str(tket_circ))

