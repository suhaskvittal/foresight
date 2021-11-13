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

from timeit import default_timer as timer
from copy import copy, deepcopy

from fs_layerview import LayerViewPass
from fs_foresight import ForeSight
from fs_stat import load_classifier
from fs_exec import _bench_and_cmp, _pad_circuit_to_fit, draw
from fs_util import _compute_per_layer_density_2q,\
                    _compute_child_distance_2q,\
                    _compute_size_depth_ratio_2q,\
                    _compute_in_layer_qubit_distance_2q
from fs_util import G_FORESIGHT_SLACK, G_FORESIGHT_SOLN_CAP

import numpy as np

from collections import defaultdict

class BenchmarkPass(AnalysisPass):
    def __init__(
        self, 
        coupling_map, 
        compare=['sabre', 'foresight', 'ssonly'],
        runs=5,
        compute_stats=False
    ):
        super().__init__()

        self.benchmark_list = compare

        self.mapping_policy = SabreLayout(coupling_map, 
            routing_pass=SabreSwap(coupling_map, heuristic='decay'), max_iterations=3)
        self.sabre_router = SabreSwap(coupling_map, heuristic='decay')
        self.foresight_router = ForeSight(
                coupling_map,
                slack=G_FORESIGHT_SLACK,
                solution_cap=G_FORESIGHT_SOLN_CAP
        )
        self.foresight_ssonly_router = ForeSight(
                coupling_map,
                slack=G_ForeSight_SLACK,
                solution_cap=1
        )
        if hybrid_data_file:
            # Declare hybrid routers.
            self.hybrid_router = MPATH_HYBRID(coupling_map, hybrid_data_file)  
            self.hybrid_pass = PassManager([
                self.hybrid_router,
                Unroller(G_QISKIT_GATE_SET)
            ])

        self.sabre_pass = PassManager([
            self.sabre_router,
            Unroller(G_QISKIT_GATE_SET)
        ])
        self.foresight_pass = PassManager([
            self.foresight_router,
            Unroller(G_QISKIT_GATE_SET)
        ])
        self.foresight_ssonly_pass = PassManager([
            self.foresight_ssonly_router,
            Unroller(G_QISKIT_GATE_SET)
        ])
        self.look_pass = PassManager([
            LookaheadSwap(coupling_map, search_depth=4, search_width=4),
            Unroller(G_QISKIT_GATE_SET)
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
        original_circuit = dag_to_circuit(dag)
        circ_cx = original_circuit.count_ops()['cx']

        self.benchmark_results = defaultdict(float)
        print('\tBenchmark Logging:')
        print('\t\tsize, depth =', dag.size(), dag.depth())

        for r in range(self.runs):
            # Get initial layout.
            circ = original_circuit.copy()
            circ = self.layout_pass.run(circ)
            # Run dag on both passes. 
            # SABRE
            if 'sabre' in self.benchmark_list:
                print('\t\t(sabre start.)')
                start = timer()
                sabre_circ = self.sabre_pass.run(circ)
                end = timer()
                sabre_cnots = sabre_circ.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['SABRE CNOTs'] > sabre_cnots:
                    self.benchmark_results['SABRE CNOTs'] = sabre_cnots
                    self.benchmark_results['SABRE Depth'] = (sabre_circ.depth())
                    self.benchmark_results['SABRE Time'] = (end - start)
                print('\t\t(sabre done.)')
            # ForeSight
            if 'foresight' in self.benchmark_list:
                print('\t\t(foresight start.)')
                start = timer()
                foresight_circ = self.foresight_pass.run(circ)
                end = timer()
                foresight_cnots = foresight_circ.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['ForeSight CNOTs'] > foresight_cnots:
                    self.benchmark_results['ForeSight CNOTs'] = foresight_cnots
                    self.benchmark_results['ForeSight Depth'] = (foresight_circ.depth())
                    self.benchmark_results['ForeSight Time'] = (end - start)
                print('\t\t(foresight done.)')
            if 'ssonly' in self.benchmark_list:
                print('\t\t(ssonly start.)')
                start = timer()
                ssonly_circ = self.foresight_ssonly_pass.run(circ)
                end = timer()
                ssonly_cnots = ssonly_circ.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['ForeSight SSOnly CNOTs'] > ssonly_cnots:
                    self.benchmark_results['ForeSight SSOnly CNOTs'] = ssonly_cnots
                    self.benchmark_results['ForeSight SSOnly Depth'] = ssonly_circ.depth()
                    self.benchmark_results['ForeSight SSOnly Time'] = end - start
                print('\t\t(ssonly done).')
            # LOOK
            if 'look' in self.benchmark_list:
                try:
                    start = timer()
                    look_circ = self.look_pass.run(circ)
                    end = timer()
                    if self.benchmark_results['LOOK CNOTs'] >= 0:
                        self.benchmark_results['LOOK CNOTs'] += (look_circ.count_ops()['cx'] - circ_cx)
                        self.benchmark_results['LOOK Depth'] += (look_circ.depth())
                        self.benchmark_results['LOOK Time'] += (end - start)
                except TranspilerError:
                    self.benchmark_results['LOOK CNOTs'] = -1.0
                    self.benchmark_results['LOOK Depth'] = -1.0
                    self.benchmark_results['LOOK Time'] = -1.0
        # Some circuit statistics as well.
        layer_view_pass = LayerViewPass()
        layer_view_pass.run(dag)
        primary_layer_view = layer_view_pass.property_set['primary_layer_view']
        if self.compute_stats:
            self.benchmark_results['Layer Density, mean'], self.benchmark_results['Layer Density, std.'] =\
                _compute_per_layer_density_2q(primary_layer_view)
            self.benchmark_results['Child Distance, mean'], self.benchmark_results['Child Distance, std.'] =\
                _compute_child_distance_2q(primary_layer_view)
            self.benchmark_results['In Layer Distance, mean'], self.benchmark_results['In Layer Distance, std.'] =\
                _compute_in_layer_qubit_distance_2q(primary_layer_view)
    
