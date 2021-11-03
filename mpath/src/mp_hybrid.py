"""

        node_to_parent = {}
        for node in primary_layer_view[0]:
            q0, q1 = node.qargs
            node_to_parent[q0] = (0, node)
            node_to_parent[q1] = (0, node)
        verified_parents = set()

    author: Suhas Vittal
    date:   26 October 2021
"""

from qiskit.circuit import Qubit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from mp_layerview import LayerViewPass
from mp_stat import load_regressor, load_regressor, get_independent_variable
from mp_util import G_MPATH_IPS_SLACK, G_MPATH_IPS_SOLN_CAP
from mp_hybrid_sabreswap import MPATH_HYBRID_SabreSwap
from mp_hybrid_ips import MPATH_HYBRID_IPS

import numpy as np

from copy import copy, deepcopy
from collections import deque, defaultdict

import warnings

class MPATH_HYBRID(TransformationPass):
    def __init__(self, coupling_map, profile_file, window_size=50):
        super().__init__()
        self.coupling_map = coupling_map
        self.regressor = load_regressor(profile_file)
        
        self.fake_run = False
        self.router_usage = defaultdict(int)
    
    def run(self, dag):
        mapped_dag = dag._copy_circuit_metadata()
        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        
        warnings.filterwarnings("ignore")

        primary_layer_view, secondary_layer_view = self.property_set['primary_layer_view'], self.property_set['secondary_layer_view']
        if primary_layer_view is None or secondary_layer_view is None:
            layer_view_pass = LayerViewPass();
            layer_view_pass.run(dag)
            primary_layer_view, secondary_layer_view = layer_view_pass.property_set['primary_layer_view'], layer_view_pass.property_set['secondary_layer_view']
        primary_layer_view = deque(primary_layer_view)
        secondary_layer_view = deque(secondary_layer_view)

        self.router_usage = defaultdict(int)
        # Get initial stats.
        X = get_independent_variable(primary_layer_view)
        y = self.regressor.predict(X)

        router = None
        if y >= 0:   # start with sabre.
            router = MPATH_HYBRID_SabreSwap(self.coupling_map, self.regressor, heuristic='decay')
        elif y < 0:
            router = MPATH_HYBRID_IPS(self.coupling_map, self.regressor, slack=G_MPATH_IPS_SLACK, solution_cap=G_MPATH_IPS_SOLN_CAP)
        mapped_dag = router.run(dag)
        self.property_set['final_layout'] = router.property_set['final_layout'] 
        self.router_usage['sabre'] = router.router_usage['sabre']
        self.router_usage['ips'] = router.router_usage['ips']
        return mapped_dag
    
