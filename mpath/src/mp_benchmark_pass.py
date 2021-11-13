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

from mp_layerview import LayerViewPass
from mp_ips import MPATH_IPS
from mp_bsp import MPATH_BSP
from mp_hybrid import MPATH_HYBRID
from mp_stat import load_classifier
from mp_exec import _bench_and_cmp, _pad_circuit_to_fit, draw
from mp_util import _compute_per_layer_density_2q,\
                    _compute_child_distance_2q,\
                    _compute_size_depth_ratio_2q,\
                    _compute_in_layer_qubit_distance_2q
from mp_util import G_MPATH_IPS_SLACK, G_MPATH_IPS_SOLN_CAP, G_MPATH_BSP_TREE_WIDTH, G_QISKIT_GATE_SET

import numpy as np

from collections import defaultdict

class BenchmarkPass(AnalysisPass):
    def __init__(self, coupling_map, hybrid_data_file, compare=['sabre', 'ips', 'ssonly', 'hybrid'], runs=5, compute_stats=False):
        super().__init__()

        self.benchmark_list = compare

        self.mapping_policy = SabreLayout(coupling_map, routing_pass=SabreSwap(coupling_map, heuristic='decay'), max_iterations=3)
        #self.mapping_policy = SabreLayout(coupling_map, routing_pass=MPATH_IPS(coupling_map, slack=G_MPATH_IPS_SLACK, solution_cap=2))
        self.sabre_router = SabreSwap(coupling_map, heuristic='decay')
        self.ips_router = MPATH_IPS(
                coupling_map,
                slack=G_MPATH_IPS_SLACK,
                solution_cap=G_MPATH_IPS_SOLN_CAP
        )
        self.ips_ssonly_router = MPATH_IPS(
                coupling_map,
                slack=G_MPATH_IPS_SLACK,
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
        self.ips_pass = PassManager([
            self.ips_router,
            Unroller(G_QISKIT_GATE_SET)
        ])
        self.ips_ssonly_pass = PassManager([
            self.ips_ssonly_router,
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
            # MPATH IPS
            if 'ips' in self.benchmark_list:
                print('\t\t(ips start.)')
                start = timer()
                ips_circ = self.ips_pass.run(circ)
                end = timer()
                ips_cnots = ips_circ.count_ops()['cx'] - circ_cx
                sabre_ips_circ = self.sabre_pass.run(ips_circ)
                if sabre_ips_circ.count_ops()['cx'] != ips_circ.count_ops()['cx']:
                    print('error: not routed correctly', sabre_ips_circ.count_ops()['cx'], ips_circ.count_ops()['cx'])
                    exit()
                if r == 0 or self.benchmark_results['MPATH_IPS CNOTs'] > ips_cnots:
                    self.benchmark_results['MPATH_IPS CNOTs'] = ips_cnots
                    self.benchmark_results['MPATH_IPS Depth'] = (ips_circ.depth())
                    self.benchmark_results['MPATH_IPS Time'] = (end - start)
                print('\t\t(ips done.)')
            if 'ssonly' in self.benchmark_list:
                print('\t\t(ssonly start.)')
                start = timer()
                ssonly_circ = self.ips_ssonly_pass.run(circ)
                end = timer()
                ssonly_cnots = ssonly_circ.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['MPATH_IPS SSOnly CNOTs'] > ssonly_cnots:
                    self.benchmark_results['MPATH_IPS SSOnly CNOTs'] = ssonly_cnots
                    self.benchmark_results['MPATH_IPS SSOnly Depth'] = ssonly_circ.depth()
                    self.benchmark_results['MPATH_IPS SSOnly Time'] = end - start
                print('\t\t(ssonly done).')
            # MPATH BSP
            if 'bsp' in self.benchmark_list:
                start = timer()
                bsp_circ = self.bsp_pass.run(circ)
                end = timer()
                bsp_cnots = bsp_circ.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['MPATH_BSP CNOTs'] > bsp_cnots:
                    self.benchmark_results['MPATH_BSP CNOTs'] = bsp_cnots
                    self.benchmark_results['MPATH_BSP Depth'] = (bsp_circ.depth())
                    self.benchmark_results['MPATH_BSP Time'] = (end - start)
                print('\t\t(bsp done.)')
            # MPATH HYBRID
            if 'hybrid' in self.benchmark_list:
                start = timer()
                hybrid_circ = self.hybrid_pass.run(circ)
                end = timer()
                hybrid_cnots = hybrid_circ.count_ops()['cx'] - circ_cx
                if r == 0 or self.benchmark_results['MPATH_HYBRID CNOTs'] > hybrid_cnots:
                    self.benchmark_results['MPATH_HYBRID CNOTs'] = hybrid_cnots
                    self.benchmark_results['MPATH_HYBRID Depth'] = (hybrid_circ.depth())
                    self.benchmark_results['MPATH_HYBRID Time'] = (end - start)
                    
                    router_usage = self.hybrid_router.router_usage
                    if router_usage['sabre'] + router_usage['ips'] == 0:
                        self.benchmark_results['MPATH_HYBRID SABRE Uptime'] = 0
                    else:
                        self.benchmark_results['MPATH_HYBRID SABRE Uptime'] = router_usage['sabre'] / (router_usage['sabre'] + router_usage['ips']) 
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
    
