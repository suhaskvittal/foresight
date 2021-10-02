"""
	author:	Suhas Vittal
	date: 	29 September 2021 @ 5:09 p.m. EST
"""

from pulp import * 

from copy import copy

def build_path_dag(coupling_map, target_set, current_layout, max_swaps):
	path_candidates = []  # Compute possible paths.
	for (v0, v1) in target_set:
		p0, p1 = current_layout[v0], current_layout[v1]
		path_list = _path_find(p0, p1, max_swaps, coupling_map)
		path_candidates.append(path_list)  # We want at least one of these paths to be followed in the path list.
	lp, var_mapping = _build_lp(path_candidates)
	status = lp.solve(GLPK(msg=False))
	if LpStatus[status] != 'Optimal':  # Terminate if we do not find an optimal solution.
		print('[ERROR] Status = %s' % LpStatus[status])
		exit()
	path_dag = [[] for _ in range(max_swaps)]
	for (i, e) in var_mapping:
		if value(var_mapping[(i,e)]) == 0.0:
			continue
		path_dag[i].append((e, value(var_mapping[(i,e)]))) 	# (edge, score)
	return path_dag

def _build_lp(path_candidates):
	# Initialize LP
	lp = LpProblem('Solver', LpMinimize)
	# Build LP Variables
	variable_mapping = dict() # store correspondence between variables and edges in coupling graph.
	n_xvar = 0
	n_zvar = 0
	n_wvar = 0
	edge_var_list = []
	edge_freq_dict = {}
	w_var_list = []
	for (_, conj_list) in enumerate(path_candidates):
		w_var = 'w%d' % n_wvar
		w = make_lp_var(w_var)
		n_wvar += 1
		z_var_list = []  # create equisatisfiable CNF formula for each DNF formula.
		for (i, conj) in enumerate(conj_list):
			z_var = 'z%d' % n_zvar
			z = make_lp_var(z_var)
			n_zvar += 1
			var_list = []
			for (j, (v1, v2)) in enumerate(conj):
				x_var = 'x%d' % n_xvar
				if (j, (v1, v2)) not in variable_mapping:
					x = make_lp_var(x_var)
					variable_mapping[(j, (v1, v2))] = x 
					edge_var_list.append(x)
					edge_freq_dict[x] = 1
					n_xvar += 1
				else:
					x = variable_mapping[(j, (v1, v2))]
					edge_freq_dict[x] += 1
				lp += (x + (1.0 - z) >= 1.0)
				var_list.append(1.0 - x)
			lp += (lpSum(var_list) + z >= 1.0)	
			lp += (w + (1.0 - z) >= 1.0)
			z_var_list.append(z)
		lp += (lpSum(z_var_list) + (1.0 - w) >= 1.0)
		w_var_list.append(w)
	lp += (lpSum(w_var_list) >= 1.0)
	# Build objective
	lp += lpSum([x for x in edge_var_list])
	return lp, variable_mapping

def _path_find(source, sink, max_length, coupling_map):
	path_list = []
	queue = [(source, [])] 
	while len(queue) > 0:
		v, prev = queue.pop(0)
		if v == sink:
			path_list.append(prev)
			continue
		if len(prev) == max_length:  # cut off long paths
			continue
		for w in coupling_map.neighbors(v):  # add neighbors
			prev_cpy = prev.copy()
			if len(prev) > 0 and (prev[-1] == (v, w) or prev[-1] == (w, v)):
				continue
			if max_length - len(prev) < coupling_map.distance_matrix[w, sink]:
				continue  # We will not reach the sink in time.
			if v < w:
				prev_cpy.append((v, w))
			else:
				prev_cpy.append((w, v))
			queue.append((w, prev_cpy))
	return path_list

def make_lp_var(name):
	return LpVariable(name, 0, 1)
