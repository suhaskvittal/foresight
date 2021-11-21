"""
    author: Suhas Vittal
    date:   24 October 2021
"""

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import *
from qiskit.exceptions import QiskitError

from timeit import default_timer as timer
from copy import copy, deepcopy

from fs_layerview import LayerViewPass
from fs_foresight import ForeSight
from fs_exec import exec_sim, total_variation_distance
from fs_util import _compute_per_layer_density_2q,\
                    _compute_child_distance_2q,\
                    _compute_size_depth_ratio_2q,\
                    _compute_in_layer_qubit_distance_2q
from fs_util import *

from jkq import qmap

import numpy as np

from collections import defaultdict
import tracemalloc
import signal
import traceback

class BenchmarkPass(AnalysisPass):
    def __init__(
        self, 
        coupling_map, 
        arch_file,
        compare=['sabre', 'foresight', 'ssonly'],
        runs=5,
        compute_stats=False,
        **kwargs
    ):
        super().__init__()

        self.basis_gates = G_QISKIT_GATE_SET
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
        self.mapping_policy = SabreLayout(coupling_map, 
            routing_pass=SabreSwap(coupling_map, heuristic='decay'), max_iterations=3)
        self.sabre_router = SabreSwap(coupling_map, heuristic='decay')
        self.foresight_router = ForeSight(
                coupling_map,
                slack=slack,
                solution_cap=solution_cap,
                debug=kwargs['debug']
        )
        self.noisy_foresight_router = ForeSight(
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
                edge_weights=edge_weights,
                vertex_weights=vertex_weights,
                readout_weights=readout_weights,
                noisy_routing=kwargs['noisy']
        )
        # Set up passes
        self.sabre_pass = PassManager([
            self.sabre_router,
            Unroller(self.basis_gates)
        ])
        self.foresight_pass = PassManager([
            self.foresight_router,
            Unroller(self.basis_gates)
        ])
        self.noisy_foresight_pass = PassManager([
            self.noisy_foresight_router,
            Unroller(self.basis_gates)
        ])
        self.foresight_ssonly_pass = PassManager([
            self.foresight_ssonly_router,
            Unroller(self.basis_gates)
        ])
        # Layout pass
        self.layout_pass = PassManager([
            self.mapping_policy,
            ApplyLayout()
        ])
        self.runs = runs
        self.compute_stats = compute_stats

        self.benchmark_results = None
    
    def run(self, dag):
        self.benchmark_results = defaultdict(float)

        original_circuit = dag_to_circuit(dag)
        circ_cx = original_circuit.count_ops()['cx']
        self.benchmark_results['Original CNOTs'] = circ_cx

        print('\tBenchmark Logging:')
        print('\t\tsize, depth =', dag.size(), dag.depth())

        signal.signal(signal.SIGALRM, lambda x, y: print('A* timed out.'))

        for r in range(self.runs):
            # Get initial layout.
            circ = original_circuit.copy()
            circ = self.layout_pass.run(circ)
            # Run dag on both passes. 
            # SABRE
            if 'sabre' in self.benchmark_list:
                print('\t\tsabre start.')
                start = timer()
                if self.measure_memory:
                    tracemalloc.start(25)
                c = self.sabre_pass.run(circ)
                if self.measure_memory:
                    ss = tracemalloc.take_snapshot()
                end = timer()
                if self.measure_memory:
                    tracemalloc.stop()
                sabre_cnots = c.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['SABRE CNOTs'] > sabre_cnots:
                    self.benchmark_results['SABRE CNOTs'] = sabre_cnots
                    self.benchmark_results['SABRE Depth'] = (c.depth())
                    self.benchmark_results['SABRE Time'] = (end - start)
                    if self.measure_memory:
                        self.benchmark_results['SABRE Memory'] = sum(stat.size for stat in ss.statistics('traceback'))/1024.0
                    # update best sabre circuit
                    sabre_circ = c
            # ForeSight
            if 'foresight' in self.benchmark_list:
                print('\t\tforesight start.')
                start = timer()
                if self.measure_memory:
                    tracemalloc.start(25)
                c = self.foresight_pass.run(circ)
                if self.measure_memory:
                    ss = tracemalloc.take_snapshot()
                end = timer()
                if self.measure_memory:
                    tracemalloc.stop()
                foresight_cnots = c.count_ops()['cx'] - circ_cx 
                if r == 0 or self.benchmark_results['ForeSight CNOTs'] > foresight_cnots:
                    self.benchmark_results['ForeSight CNOTs'] = foresight_cnots
                    self.benchmark_results['ForeSight Depth'] = (c.depth())
                    self.benchmark_results['ForeSight Time'] = (end - start)
                    if self.measure_memory:
                        self.benchmark_results['ForeSight Memory'] = sum(stat.size for stat in ss.statistics('traceback'))/1024.0
                    foresight_circ = c
                if self.noise_model:    
                    print('\t\tnoisy foresight start.')
                    start = timer()
                    c = self.noisy_foresight_pass.run(circ)
                    end = timer()
                    noisy_foresight_cnots = c.count_ops()['cx'] - circ_cx
                    if r == 0 or self.benchmark_results['Noisy ForeSight CNOTs'] > noisy_foresight_cnots:
                        self.benchmark_results['Noisy ForeSight CNOTs'] = noisy_foresight_cnots
                        self.benchmark_results['Noisy ForeSight Depth'] = c.depth()
                        self.benchmark_results['Noisy ForeSight Time'] = (end - start)
                    noisy_foresight_circ = c
            if 'ssonly' in self.benchmark_list:
                print('\t\tssonly start.')
                start = timer()
                if self.measure_memory:
                    tracemalloc.start(25)
                c = self.foresight_ssonly_pass.run(circ)
                if self.measure_memory:
                    ss = tracemalloc.take_snapshot()
                end = timer()
                if self.measure_memory:
                    tracemalloc.stop()
                ssonly_cnots = c.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['ForeSight SSOnly CNOTs'] > ssonly_cnots:
                    self.benchmark_results['ForeSight SSOnly CNOTs'] = ssonly_cnots
                    self.benchmark_results['ForeSight SSOnly Depth'] = c.depth()
                    self.benchmark_results['ForeSight SSOnly Time'] = end - start
                    if self.measure_memory:
                        self.benchmark_results['ForeSight SSOnly Memory'] = sum(stat.size for stat in ss.statistics('traceback'))/1024.0
                    ssonly_circ = c
            if 'a*' in self.benchmark_list:
                print('\t\ta* start.')
                # Collect results from A* search
                if r == 0:
                    self.benchmark_results['A* CNOTs'] = 0  # init in case we skip A* or an error occurs
                    self.benchmark_results['A* Depth'] = 0
                    self.benchmark_results['A* Time'] = 0
                    self.benchmark_results['A* Memory'] = 0
                try:
                    qmap_pass = PassManager([
                        Unroller(['u1', 'u2', 'u3', 'id', 'u', 'p', 'cx'])
                    ])
                    start = timer()
                    if self.measure_memory:
                        tracemalloc.start(25)
                    signal.alarm(60)  # A* can deadlock itself
                    qmap_res = qmap.compile(
                        circ,
                        arch=self.arch_file,
                        method=qmap.Method.heuristic,
                        initial_layout=qmap.InitialLayoutStrategy.identity
                    )
                    signal.alarm(0)
                    if self.measure_memory:
                        ss = tracemalloc.take_snapshot()
                    end = timer()
                    if self.measure_memory:
                        tracemalloc.stop()
                    # Benchmark A* search
                    c = QuantumCircuit.from_qasm_str(qmap_res['mapped_circuit']['qasm'])
                    c = qmap_pass.run(c)  # Compile gates to basis gates.
                    if r == 0 or self.benchmark_results['A* CNOTs'] > c.count_ops()['cx'] - circ_cx:
                        self.benchmark_results['A* CNOTs'] = c.count_ops()['cx'] - circ_cx
                        self.benchmark_results['A* Depth'] = c.depth()
                        self.benchmark_results['A* Time'] = end - start
                        qmap_circ = c
                    if self.measure_memory:
                        self.benchmark_results['A* Memory'] = sum(stat.size for stat in ss.statistics('traceback'))/1024.0
                except (QiskitError, KeyError) as error:
                    print('\t\t(a* failure)')
                    traceback.print_exc()
        if self.simulate:
            # original
            ideal_counts = exec_sim(circ, basis_gates=self.basis_gates) 
            # SABRE
            sabre_counts = exec_sim(sabre_circ, basis_gates=self.basis_gates, noise_model=self.noise_model) 
            self.benchmark_results['SABRE TVD'] = total_variation_distance(ideal_counts, sabre_counts)
            # Foresight
            foresight_counts = exec_sim(foresight_circ, basis_gates=self.basis_gates, noise_model=self.noise_model) 
            self.benchmark_results['ForeSight TVD'] = total_variation_distance(ideal_counts, foresight_counts)
            # Noisy ForeSight
            noisy_foresight_counts = exec_sim(noisy_foresight_circ, basis_gates=self.basis_gates, noise_model=self.noise_model)
            self.benchmark_results['Noisy ForeSight TVD'] = total_variation_distance(ideal_counts, noisy_foresight_counts)
            # Relative
            self.benchmark_results['SABRE Relative TVD'] = self.benchmark_results['Noisy ForeSight TVD'] / self.benchmark_results['SABRE TVD']
            self.benchmark_results['ForeSight Relative TVD'] = self.benchmark_results['Noisy ForeSight TVD'] / self.benchmark_results['ForeSight TVD']
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
    
