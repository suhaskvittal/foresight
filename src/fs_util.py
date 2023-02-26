"""
    author: Suhas Vittal
    date:   20 October 2021
"""

from qiskit.transpiler import CouplingMap

import numpy as np
import pickle as pkl

from os import listdir
from os.path import isfile, join

def read_arch_file(arch_file):
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

# GATE SETS
G_QISKIT_GATE_SET = ['cx', 'rz', 'sx', 'x', 'id']
#G_QISKIT_GATE_SET = ['cx', 'u3']

# ALGORITHM PARAMETERS
G_FORESIGHT_SOLN_CAP = 32
G_FORESIGHT_SLACK = 2

def get_error_rates_from_ibmq_backend(ibmq_backend):
    coupling_map = CouplingMap(ibmq_backend.configuration().coupling_map)

    sq_error_rates = [0 for _ in range(coupling_map.size())]
    cx_error_rates = {}
    ro_error_rates = [0 for _ in range(coupling_map.size())]

    mean_coh_t1 = 0
    mean_cx_time = 0

    for p in coupling_map.physical_qubits:
        error = 0
        for g in G_QISKIT_GATE_SET:
            if g == 'cx' or g == 'ecr':
                continue
            e = ibmq_backend.properties().gate_error(g, p)
            if e > error:
                error = e
        sq_error_rates[p] = error
        ro_error_rates[p] = ibmq_backend.properties().readout_error(p)
        try:
            mean_coh_t1 += ibmq_backend.properties().t1(p) * 1e9
        except:
            mean_coh_t1 += 0.0
    for edge in coupling_map.get_edges():
        e0, e1  = tuple(edge)
        try:
            cx_error_rates[(e0, e1)] = ibmq_backend.properties().gate_error('cx', edge)
            mean_cx_time += ibmq_backend.properties().gate_length('cx', edge) * 1e9
        except:
            cx_error_rates[(e0, e1)] = ibmq_backend.properties().gate_error('ecr', edge)
            mean_cx_time += ibmq_backend.properties().gate_length('ecr', edge) * 1e9
        cx_error_rates[(e1, e0)] = cx_error_rates[(e0, e1)]
    mean_coh_t1 /= coupling_map.size()
    mean_cx_time /= len(coupling_map.get_edges())
    return sq_error_rates, cx_error_rates, ro_error_rates, mean_coh_t1, mean_cx_time

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
    
