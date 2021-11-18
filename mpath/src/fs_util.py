"""
    author: Suhas Vittal
    date:   20 October 2021
"""

from qiskit.transpiler import CouplingMap

import numpy as np
import pickle as pkl

from os import listdir
from os.path import isfile, join

def _get_qasm_files(folder):
    return folder, [
        f for f in listdir(folder)\
        if isfile(join(folder, f)) and f.endswith('.qasm')
    ]

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
G_IBM_TOKYO = _read_arch_file('arch/ibm_tokyo.arch')
G_100GRID = _read_arch_file('arch/100grid.arch')

# GATE SETS
G_QISKIT_GATE_SET = ['u1', 'u2', 'u3', 'cx']

# BENCHMARK SUITES
G_QASMBENCH_MEDIUM = [
    'adder_n10',        # single adder
#    'qft_n15',
#    'dnn_n8',           # quantum deep neural net
#    'cc_n12',           # counterfeit coin
    'multiply_n13',
    'multiplier_n15',   # binary multiplier
#    'qf21_n15',         # quantum phase estimation, factor 21           
    'sat_n11',      
    'seca_n11',         # shor's error correction
    'bv_n14',           # bernstein-vazirani algorithm 
#    'ising_n10',        # ising gate sim
#    'qaoa_n6',          
#    'qpe_n9',           # quantum phase estimation
#    'simon_n6',         # simon's algorithm 
#    'vqe_uccsd_n6',
#    'vqe_uccsd_n8'
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
G_ZULEHNER = _get_qasm_files('benchmarks/zulehner')
G_ZULEHNER_PARTIAL = _get_qasm_files('benchmarks/zulehner_partial')
G_QAOA_SK = _get_qasm_files('benchmarks/qaoa_sk')
G_QAOA_3RL = _get_qasm_files('benchmarks/qaoa_3r_large')
G_QAOA_3RVL = _get_qasm_files('benchmarks/qaoa_3r_vlarge')
G_BV_VL = _get_qasm_files('benchmarks/bv_vlarge')
G_BV_8to15 = _get_qasm_files('benchmarks/bv_8-15')

# ALGORITHM PARAMETERS
G_FORESIGHT_SOLN_CAP = 32
G_FORESIGHT_SLACK = 3

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
    
WEIGHT_MAX = 5

def _compute_per_layer_density_2q(primary_layer_view, weighted=False):
    if len(primary_layer_view)==0:
        return 0, 0
    densities = []
    for (i, layer) in enumerate(primary_layer_view):
        if weighted:
            if i <= WEIGHT_MAX:
                densities.append(len(layer))
        else:
            densities.append(len(layer))
    return np.mean(densities), np.std(densities)

def _compute_child_distance_2q(primary_layer_view, weighted=False):
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
                        if weighted:
                            if num_layers <= WEIGHT_MAX:
                                child_distances.append(num_layers - home_layer)
                        else:
                            child_distances.append(num_layers - home_layer)
                        verified_parents.add(parent)
                node_to_parent[q] = (num_layers, child)
            has_2q_ops = True
        num_layers += 1 if has_2q_ops else 0
    if len(child_distances) == 0:
        return 0, 0
    else:
        return np.mean(child_distances), np.std(child_distances)

def _compute_size_depth_ratio_2q(primary_layer_view):
    return sum(len(layer) for layer in primary_layer_view) / len(primary_layer_view)

def _compute_in_layer_qubit_distance_2q(primary_layer_view, weighted=False):
    distances = []
    for (i, layer) in enumerate(primary_layer_view):
        used_qubits_left = []
        used_qubits_right = []
        for node in layer:
            q0, q1 = node.qargs
            used_qubits_left.append(q0.index)
            used_qubits_right.append(q1.index)
        d = 0
        for ii in range(len(used_qubits_left)):
            q0, q1 = used_qubits_left[ii], used_qubits_right[ii]
            for jj in range(ii+1, len(used_qubits_left)):
                r0, r1 = used_qubits_left[jj], used_qubits_right[jj] 
                d += abs(q0-r0)+abs(q0-r1)+abs(q1-r0)+abs(q1-r1)
        if weighted:
            if i <= WEIGHT_MAX:
                distances.append(d)
        else:
            distances.append(d)
    return np.mean(distances), np.std(distances)
    
