"""
    author: Suhas Vittal
    date:   20 October 2021
"""

from qiskit.transpiler import CouplingMap

import numpy as np
import pickle as pkl

from os import listdir
from os.path import isfile, join

# DATA COLLECTION FUNCTIONS
def _sk_benchmarks():
    sk_file = 'SK.pkl'
    export_data = None
    with open(sk_file, 'rb') as reader:
        export_data = pkl.load(reader)
    circs = []
    for family in export_data:
        for grid_search_type in ['grid_search_30_30', 'grid_search_60_60']:
            circs.append((family, grid_search_type, export_data[family][grid_search_type]['opt_circ'])) 
    return circs

def _read_arch_file(arch_file):
    reader = open(arch_file, 'r')
    # Don't need first line.
    reader.readline()
    
    edges = []
    line = reader.readline()
    while line != '':
        split_line = line.split(' ')
        v1, v2 = int(split_line[0]), int(split_line[1])
        edges.append([v1, v2])
        line = reader.readline()
    reader.close()
    return CouplingMap(edges)

# COUPLING MAPS
G_IBM_TORONTO = _read_arch_file('arch/ibm_toronto.arch')
G_GOOGLE_WEBER = _read_arch_file('arch/google_weber.arch')
G_RIGETTI_ASPEN9 = _read_arch_file('arch/rigetti_aspen9.arch')

# GATE SETS
G_QISKIT_GATE_SET = ['u1', 'u2', 'u3', 'cx']

# BENCHMARK SUITES
G_QASMBENCH_MEDIUM = [
    'adder_n10',        # single adder
    'qft_n15',
    'dnn_n8',           # quantum deep neural net
    'cc_n12',           # counterfeit coin
    'multiplier_n15',   # binary multiplier
    'qf21_n15',         # quantum phase estimation, factor 21           
    'sat_n11',      
    'seca_n11',         # shor's error correction
    'bv_n14',           # bernstein-vazirani algorithm 
    'ising_n10',        # ising gate sim
    'qaoa_n6',          
    'qpe_n9',           # quantum phase estimation
    'simon_n6',         # simon's algorithm 
    'vqe_uccsd_n6',
    'vqe_uccsd_n8'
]
G_QASMBENCH_LARGE = [
    'bigadder_n18',     # ripple carry adder    
#    'qft_n20',
    'ising_n26',        # ising gate sim
    'bv_n19',           # bernstein-vazirani algorithm  
    'dnn_n16',          # quantum deep neural net
    'multiplier_n25',   # binary multiplier
    'wstate_n27',       
    'ghz_state_n23',
    'cat_state_n22',    
    'square_root_n18',  # square root   
    'cc_n18'            # counterfeit coin
]
G_ZULEHNER = [f for f in listdir('benchmarks/zulehner') if isfile(join('benchmarks/zulehner', f)) and f.endswith('.qasm')]

G_QAOA = _sk_benchmarks()

# ALGORITHM PARAMETERS
G_MPATH_IPS_SOLN_CAP = 16
G_MPATH_IPS_SLACK = 2
G_MPATH_BSP_TREE_WIDTH = 32

def _soln_hash_f(soln):
    h = 0
    PRIME = 5586537595543
    for (i, layer) in enumerate(soln):
        for (p0, p1) in layer:
            h += ((2**p0)*(3**p1)*(5**i)) % PRIME
    return h

def _path_hash_f(path):
    h = 0
    PRIME = 5586537595543
    for (p0, p1) in path:
        h += ((2**p0)*(3**p1)) % PRIME
    return h    

def _path_to_swap_collection(path):
    collection = []
    for (v1, v2) in path:
        collection.append([(v1, v2)])
    return collection

def _is_pow2(x):
    return (x & (x-1)) == 0
    
def _compute_per_layer_density_2q(primary_layer_view):
    if len(primary_layer_view)==0:
        return 0, 0
    densities = []
    for layer in primary_layer_view:
        densities.append(len(layer))
    return np.mean(densities), np.std(densities)

def _compute_child_distance_2q(primary_layer_view):
    if len(primary_layer_view)==0:
        return 0, 0
    node_to_parent = {}
    for node in primary_layer_view[0]:
        if node.type == 'op' and len(node.qargs) == 2:
            q0, q1 = node.qargs
            node_to_parent[q0] = (0, node)
            node_to_parent[q1] = (0, node)
    verified_parents = set()
    child_distances = []
    num_layers = 0
    for layer in primary_layer_view:
        if num_layers == 0:
            num_layers += 1
            continue
        has_2q_ops = False
        for child in layer:
            q0, q1 = child.qargs
            for q in child.qargs:
                if q in node_to_parent:
                    home_layer, parent = node_to_parent[q]
                    if parent not in verified_parents:
                        child_distances.append(num_layers - home_layer)
                        verified_parents.add(parent)
                node_to_parent[q] = (num_layers, child)
            has_2q_ops = True
        num_layers += 1 if has_2q_ops else 0
    if len(child_distances) == 0:
        return 0, 0
    else:
        return np.mean(child_distances), np.std(child_distances)
