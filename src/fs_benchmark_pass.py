"""
    author: Suhas Vittal
    date:   24 October 2021
"""

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import *
from qiskit.exceptions import QiskitError

from fs_layerview import LayerViewPass
from fs_foresight import ForeSight
from fs_exec import exec_sim, total_variation_distance
from fs_statsabre import StatSABRE
from fs_util import _compute_per_layer_density_2q,\
                    _compute_child_distance_2q,\
                    _compute_size_depth_ratio_2q,\
                    _compute_in_layer_qubit_distance_2q
from fs_util import *

from jkq import qmap
from pytket import Circuit as tket_Circuit
from pytket.circuit import Qubit as tket_Qubit
from pytket.circuit import Node as tket_Node
from pytket.qasm import circuit_from_qasm_str as tket_q2c
from pytket.qasm import circuit_to_qasm_str as tket_c2q
from pytket.passes import RoutingPass as tket_RoutingPass
from pytket.passes import CXMappingPass as tket_CXMappingPass
from pytket.routing import Placement as tket_Placement
from pytket.routing import Architecture as tket_Arch

import numpy as np

from collections import defaultdict
from timeit import default_timer as timer
from copy import copy, deepcopy
import tracemalloc
import signal
import traceback
import pickle as pkl

class BenchmarkPass(AnalysisPass):
    def __init__(
        self, 
        coupling_map, 
        arch_file,
        compare=['sabre', 'foresight', 'foresight_ssonly'],
        runs=5,
        compute_stats=False,
        **kwargs
    ):
        """
            Qiskit Pass for benchmarking. We benchmark according to 
            Qiskit optimization level 3. Each run on a circuit yields
            a mapping from SabreLayout -- the lowest SWAP outcome on
            all runs is recorded.

            coupling_map: backend connection structure
            arch_file: file for backend structure (needed for A* search)
            compare: list of passes to benchmark on
                possible values:
                    sabre
                    foresight
                    foresight_noisy
                    foresight_ssonly (one solution ForeSight, ignores kwargs['solncap']),
                    foresight_dynamic (experimental new version of ForeSight),
                    tket,
                    a*

            runs: runs per circuit
            kwargs: 
                sim: True if we want to run a simulation of each circuit (collect shots).
                noisy: True if we want to perform noisy routing (and simulation if sim is True)
                slack: the constraint relaxation factor for ForeSight
                solncap: the maximum number of solutions in ForeSight's computation tree
                debug: send out debug messages in ForeSight
        """
        super().__init__()

        self.basis_gates = G_QISKIT_GATE_SET
        self.coupling_map = coupling_map
        self.arch_file = arch_file

        # Parse kwargs
        self.simulate = kwargs['sim']
        self.measure_memory = kwargs['mem']
        # Set up noise model
        if kwargs['noisy']:
            self.noise_model, edge_weights, vertex_weights, readout_weights = kwargs['noise_model']
        else:
            self.noise_model = None
            edge_weights, vertex_weights, readout_weights = None,None,None
        slack = kwargs['slack']
        solution_cap = kwargs['solncap']

        # Set up routers
        self.benchmark_list = compare
        if self.noise_model:
            self.benchmark_list.append('foresight_noisy')

        # Set up routing passes
        self.sabre_router = StatSABRE(coupling_map, heuristic='decay')
        self.foresight_router = ForeSight(
                coupling_map,
                slack=slack,
                solution_cap=solution_cap,
                debug=kwargs['debug']
        )
        self.foresight_noisy_router = ForeSight(
                coupling_map,
                slack=slack,
                solution_cap=solution_cap,
                debug=kwargs['debug'],
                edge_weights=edge_weights
        )
        self.foresight_ssonly_router = ForeSight(
                coupling_map,
                slack=slack,
                solution_cap=1,
                debug=kwargs['debug'],
                edge_weights=edge_weights
        )
        self.foresight_d_router = ForeSight(
                coupling_map,
                slack=slack,
                solution_cap=solution_cap,
                asap_boost=True,
                approx_asap=True,
                debug=kwargs['debug']
        )
#        self.foresight_asap_router = ForeSight(
#                coupling_map,
#                slack=slack,
#                solution_cap=solution_cap//2,
#                asap_boost=True,
#                asap_only=True,
#                debug=kwargs['debug']
#        )
        self.benchmark_passes = {
            'sabre': ('SABRE', postrouting_qiskitopt3(coupling_map, routing_pass=self.sabre_router)), 
            'foresight': ('ForeSight', postrouting_qiskitopt3(coupling_map, routing_pass=self.foresight_router)),
            'foresight_ssonly': ('ForeSight SSOnly', postrouting_qiskitopt3(coupling_map, routing_pass=self.foresight_ssonly_router)),
            'foresight_noisy': ('Noisy ForeSight', postrouting_qiskitopt3(coupling_map, routing_pass=self.foresight_noisy_router)),
            'foresight_dynamic': ('ForeSight-D', postrouting_qiskitopt3(coupling_map, routing_pass=self.foresight_d_router)),
#            'foresight_asap': ('Foresight-ASAP', postrouting_qiskitopt3(coupling_map, routing_pass=self.foresight_asap_router)),
            'a*': ('A*', None),
            'tket': ('TKET', None)
        }
        self.prerouting_pass = prerouting_qiskitopt3(coupling_map)

        self.runs = runs
        self.compute_stats = compute_stats

        self.benchmark_results = None
        self.simulation_counts = None
    
    def run(self, dag):
        self.benchmark_results = defaultdict(float)

        original_circuit = dag_to_circuit(dag)

        print('\tBenchmark Logging:')
        print('\t\tsize, depth =', dag.size(), dag.depth())

        signal.signal(signal.SIGALRM, lambda x, y: print('A* timed out.'))

        best_circuits = {}

        for r in range(self.runs):
            # Get initial layout.
            circ = original_circuit.copy()
            circ = self.prerouting_pass.run(circ)
            circ_cx = circ.count_ops()['cx']
            if r == 0:
                self.benchmark_results['Original CNOTs'] = circ_cx
            # Run dag on both passes. 
            # SABRE
            for policy in self.benchmark_list:
                name, routing_pass = self.benchmark_passes[policy]
                print('\t\t%s start.' % name)
                start = timer()
                if self.measure_memory:
                    tracemalloc.start(25)
                if policy == 'a*':
                    # Collect results from A* search
                    if r == 0:
                        self.benchmark_results['A* CNOTs'] = 0  # init in case we skip A* or an error occurs
                        self.benchmark_results['A* Depth'] = 0
                        self.benchmark_results['A* Time'] = 0
                        self.benchmark_results['A* Memory'] = 0
                    try:
                        signal.alarm(60)  # A* can deadlock itself
                        qmap_res = qmap.compile(
                            circ,
                            arch=self.arch_file,
                            method=qmap.Method.heuristic,
                            initial_layout=qmap.InitialLayoutStrategy.identity
                        )
                        signal.alarm(0)
                        # Benchmark A* search
                        c = QuantumCircuit.from_qasm_str(qmap_res['mapped_circuit']['qasm'])
                    except (QiskitError, KeyError) as error:
                        print('\t\t(A* failure)')
                        c = None
                        traceback.print_exc()
                elif policy == 'tket':
                    tket_arch = tket_Arch(self.coupling_map.get_edges())
                    trivial_tket_map = {tket_Qubit(i):tket_Node(i) for i in range(circ.num_qubits)}
                    tket_placement = tket_Placement(tket_arch)
                    tket_circuit = tket_q2c(circ.qasm())
                    tket_Placement.place_with_map(tket_circuit, trivial_tket_map)
                    tket_RoutingPass(tket_arch).apply(tket_circuit)
                    tket_CXMappingPass(tket_arch, tket_placement).apply(tket_circuit)
                    c = QuantumCircuit.from_qasm_str(tket_c2q(tket_circuit))
                else:
                    c = routing_pass.run(circ)
                # Measure number of consolidated blocks due to Qiskit optimization
                self.consolidated_blocks = 0
                # Define callback function
                def cb(pass_, dag, **kwargs):
                    pass_name = type(pass_).__name__ 
#                    if pass_name == 'ConsolidateBlocks' or pass_name == 'UnitarySynthesis' or pass_name == 'ContainsInstruction':
#                        print('\t\t\t%s' % pass_name, dag.count_ops())
                    if pass_name == 'ConsolidateBlocks' and 'unitary' in dag.count_ops():
                        self.consolidated_blocks += dag.count_ops()['unitary']
                # Transpile circuit
                if c is None:
                    continue
                c = transpile(
                    c,
                    basis_gates=G_QISKIT_GATE_SET,
                    coupling_map=self.coupling_map,
                    layout_method='trivial',
                    routing_method='none',
                    optimization_level=3,
                    approximation_degree=1,
                    callback=cb
                )
                if self.measure_memory:
                    ss = tracemalloc.take_snapshot()
                end = timer()
                if self.measure_memory:
                    tracemalloc.stop()
                cnots = c.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['%s CNOTs' % name] > cnots:
                    self.benchmark_results['%s CNOTs' % name] = cnots
                    self.benchmark_results['%s Depth' % name] = (c.depth())
                    self.benchmark_results['%s Time' % name] = (end - start)
                    if self.measure_memory:
                        self.benchmark_results['%s Memory' % name] = sum(stat.size for stat in ss.statistics('traceback'))/1024.0
                    if name == 'ForeSight-D':
                        self.benchmark_results['ForeSight-D ALAP Proportion'] = self.foresight_d_router.alap_used
                        self.benchmark_results['ForeSight-D ASAP Proportion'] = self.foresight_d_router.asap_used
                    self.benchmark_results['%s Consolidated Blocks' % name] = self.consolidated_blocks
                    best_circuits[policy] = c
        if self.simulate:
            # original
            ideal_counts = exec_sim(circ, basis_gates=self.basis_gates) 
            # SABRE
            sabre_counts = exec_sim(best_circuits['sabre'], basis_gates=self.basis_gates, noise_model=self.noise_model) 
            self.benchmark_results['SABRE TVD'] = total_variation_distance(ideal_counts, sabre_counts)
            # Foresight
            foresight_counts = exec_sim(best_circuits['foresight_dynamic'], basis_gates=self.basis_gates, noise_model=self.noise_model) 
            self.benchmark_results['ForeSight TVD'] = total_variation_distance(ideal_counts, foresight_counts)
            print(foresight_counts, ideal_counts)
            # Noisy ForeSight
            if self.noise_model:
                noisy_foresight_counts = exec_sim(best_circuits['foresight_noisy'], basis_gates=self.basis_gates, noise_model=self.noise_model)
                self.benchmark_results['Noisy ForeSight TVD'] = total_variation_distance(ideal_counts, noisy_foresight_counts)
                # Relative
                self.benchmark_results['SABRE Relative TVD'] = self.benchmark_results['Noisy ForeSight TVD']\
                                                                        / self.benchmark_results['SABRE TVD']
                self.benchmark_results['ForeSight Relative TVD'] = self.benchmark_results['Noisy ForeSight TVD']\
                                                                        / self.benchmark_results['ForeSight TVD']
            # Save counts
            counts_dict = {
                'sabre': sabre_counts,
                'foresight_dynamic': foresight_counts,
            }
            if self.noise_model:
                counts_dict['noisy foresight'] = noisy_foresight_counts 
            self.simulation_counts = counts_dict

        # Some circuit statistics as well.
        if self.compute_stats:
            layer_view_pass = LayerViewPass()
            layer_view_pass.run(dag)
            primary_layer_view = layer_view_pass.property_set['primary_layer_view']
            # Compute stats
            self.benchmark_results['Layer Density, mean'], self.benchmark_results['Layer Density, std.'] =\
                _compute_per_layer_density_2q(primary_layer_view)
            self.benchmark_results['Child Distance, mean'], self.benchmark_results['Child Distance, std.'] =\
                _compute_child_distance_2q(primary_layer_view)
            self.benchmark_results['In Layer Distance, mean'], self.benchmark_results['In Layer Distance, std.'] =\
                _compute_in_layer_qubit_distance_2q(primary_layer_view)
    
def prerouting_qiskitopt3(coupling_map):
    _unroll3q = [
        UnitarySynthesis(
            G_QISKIT_GATE_SET,
            approximation_degree=1,
            coupling_map=coupling_map
        ),
        Unroll3qOrMore()
    ]
    _reset_and_meas = [
        RemoveResetInZeroState(),
        OptimizeSwapBeforeMeasure(),
        RemoveDiagonalGatesBeforeMeasure()
    ]
    _layout = [SabreLayout(coupling_map, routing_pass=SabreSwap(coupling_map, heuristic='lookahead'), max_iterations=3)]
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]
    _swap_check = [CheckMap(coupling_map)]
    m = PassManager()
    m.append(_unroll3q)
    m.append(_reset_and_meas)
    m.append(_layout)
    m.append(_embed)
    m.append(_swap_check)
    return m


def postrouting_qiskitopt3(coupling_map, routing_pass=None, basis_gates=G_QISKIT_GATE_SET):
    _swap = [BarrierBeforeFinalMeasurements()]
    if routing_pass:
        _swap.append(routing_pass)
    def _swap_condition(prop_set):
        return not prop_set['is_swap_mapped']
    _unroller = [Unroller(basis_gates)]
    _reset = [RemoveResetInZeroState()]
    _depth_check = [Depth(), FixedPoint("depth")]
    _opt = [
        Collect2qBlocks(),
        ConsolidateBlocks(basis_gates=basis_gates),
        UnitarySynthesis(
            basis_gates,
            approximation_degree=1,
            coupling_map=coupling_map,
        ),
        Optimize1qGatesDecomposition(basis_gates),
        CommutativeCancellation(),
    ]

    def _opt_control(prop_set):
        return not prop_set['depth_fixed_point']

    m = PassManager()
#    m.append(_depth_check + _opt + _unroller, do_while=_opt_control)
#    if routing_pass:
#        m.append(_swap, condition=_swap_condition)
#    else:
#        m.append(_swap)
#    m.append(_unroller)
#    m.append(_reset)
#    m.append(_depth_check + _opt + _unroller, do_while=_opt_control)
    m.append(_swap)
    return m


