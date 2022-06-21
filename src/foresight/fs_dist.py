"""
    author: Suhas Vittal
    date:   25 October 2021
"""

from qiskit.circuit import QuantumRegister
from qiskit.transpiler.layout import Layout

from foresight.fs_hashed_layout import HashedLayout

import numpy as np

def process_coupling_map(coupling_map, path_slack, edge_weights=None):
    dist_array, next_array = _floyd_warshall(
        coupling_map,
        edge_weights=edge_weights,
    )   
    
    tmp_qreg = QuantumRegister(coupling_map.size())
    trivial_layout_dict = {tmp_qreg[p]:p for p in coupling_map.physical_qubits}
    
    path_memoizer = {}
    paths_on_arch = {}
    for p0 in coupling_map.physical_qubits:
        for p1 in coupling_map.physical_qubits:
            paths = _get_all_paths(
                p0,
                p1,
                coupling_map,
                path_slack,
                dist_array,
                next_array,
                path_memoizer,
                edge_weights=edge_weights,
            )
            layout_table = {}
            for path in paths:
                test_layout = Layout(input_dict=trivial_layout_dict)
                score = 0
                for (r0,r1) in path:
                    test_layout.swap(r0,r1)
                    if edge_weights is None:
                        score += 1.0
                    else:
                        score += edge_weights[(r0,r1)]
                hashed_layout = HashedLayout.from_layout(test_layout)
                if hashed_layout not in layout_table:
                    layout_table[hashed_layout] = (path, score)
                else:
                    _, min_score = layout_table[hashed_layout]
                    if score < min_score:
                        layout_table[hashed_layout] = (path, score)
            paths_on_arch[(p0,p1)] = [layout_table[hl][0] for hl in layout_table]

    return dist_array, paths_on_arch    

def _floyd_warshall(coupling_map, edge_weights=None):
    n = coupling_map.size()
    dist_array = [[np.infty for _ in range(n)] for _ in range(n)]
    next_array = [[None for _ in range(n)] for _ in range(n)]
    # Perform initialization
    for (p0, p1) in coupling_map.get_edges():
        dist_array[p0][p1] = 1.0 if edge_weights is None else edge_weights[(p0, p1)]
        next_array[p0][p1] = p1
    for p in coupling_map.physical_qubits:
        dist_array[p][p] = 0.0
        next_array[p][p] = p
    # DP step.
    for k in range(n):
        for i in range(n):
            for j in range(n):  
                if dist_array[i][j] > dist_array[i][k] + dist_array[k][j]:
                    dist_array[i][j] = dist_array[i][k] + dist_array[k][j]
                    next_array[i][j] = next_array[i][k]
    return dist_array, next_array

def _get_path(source, sink, next_array):
    if next_array[source][sink] is None:
        # The vertices are disconnected.
        return []
    path = []
    curr = source
    while curr != sink:
        nxt = next_array[curr][sink]
        path.append((curr, nxt))
        curr = nxt
    return path

def _get_all_paths(
    source, 
    sink, 
    coupling_map, 
    slack, 
    dist_array, 
    next_array, 
    path_memoizer, 
    edge_weights=None
):
    if (source, sink) in path_memoizer:
        shortest_path = path_memoizer[(source, sink)]
    else:
        shortest_path = _get_path(source, sink, next_array)
        path_memoizer[(source, sink)] = shortest_path
    paths = [shortest_path]
    if slack < 0 or len(shortest_path) == 0:
        return paths  # We are done. 
    neighbor_used_in_path = shortest_path[0][1]
    antineighbor_used_in_path = shortest_path[-1][0]
    for p in coupling_map.neighbors(source):
        if p == neighbor_used_in_path:
            continue
        base_edge = (source, p)
        edge_length = 1.0 if edge_weights is None else edge_weights[base_edge]
        lhs = dist_array[p][sink] + edge_length
        rhs = dist_array[source][sink] + slack
        if lhs <= rhs:
            incomplete_slack_paths = _get_all_paths(
                p,
                sink,
                coupling_map,
                slack - lhs,
                dist_array,
                next_array,
                path_memoizer,
                edge_weights=edge_weights
            )
            for path in incomplete_slack_paths:
                base_path = [base_edge]
                base_path.extend(path)
                paths.append(base_path)
#    for p in coupling_map.neighbors(sink):
#        if p == antineighbor_used_in_path:
#            continue
#        base_edge = (p, sink)
#        edge_length = 1.0 if edge_weights is None else edge_weights[base_edge]
#        lhs = dist_array[source][p] + edge_length
#        rhs = dist_array[source][sink] + slack
#        if lhs <= rhs:
#            incomplete_slack_paths = _get_all_paths(
#                source,
#                p,
#                coupling_map,
#                slack - lhs,
#                dist_array,
#                next_array,
#                path_memoizer,
#                edge_weights=edge_weights
#            )
#            for path in incomplete_slack_paths:
#                path_copy = path.copy()
#                path_copy.append(base_edge)
#                paths.append(path_copy)
    return paths

