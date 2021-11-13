"""
	author: Suhas Vittal
	date:	25 October 2021
"""

import numpy as np

def process_coupling_map(coupling_map, path_slack, edge_weights=None):
	dist_array, next_array = _floyd_warshall(coupling_map, edge_weights=edge_weights)	
	
	path_memoizer = {}
	paths_on_arch = {}
	for p0 in coupling_map.physical_qubits:
		for p1 in coupling_map.physical_qubits:
			paths_on_arch[(p0, p1)] = _get_all_paths(
				p0,
				p1,
				coupling_map,
				path_slack,
				dist_array,
				next_array,
				path_memoizer,
				edge_weights=edge_weights
			)
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
	if slack <= 0 or len(shortest_path) == 0:
		return paths  # We are done. 
	neighbor_used_in_path = shortest_path[0][1]
	for p in coupling_map.neighbors(source):
		if p == neighbor_used_in_path:
			continue
		base_edge = (source, p)
		edge_length = 1.0 if edge_weights is None else edge_weights[base_edge]
		if dist_array[p][sink] + edge_length <= dist_array[source][sink] + slack:
			incomplete_slack_paths = _get_all_paths(
				p,
				sink,
				coupling_map,
				slack - (dist_array[p][sink] + edge_length - dist_array[source][sink]),
				dist_array,
				next_array,
				path_memoizer,
				edge_weights=edge_weights
			)
			for path in incomplete_slack_paths:
				base_path = [base_edge]
				base_path.extend(path)
				paths.append(base_path)
	return paths

