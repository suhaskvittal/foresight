"""
	author: Suhas Vittal
	date:	29 September 2021 @ 5:06 p.m. EST
"""

import numpy as np

from copy import copy, deepcopy

R_SKIP_TH = 0.5

def compute_candidate(path_dag):
	soln_layers = []		# the candidate 
	soln_size = 0		  	# for solution selection
	swap_score_list = []	# for score update
	
	prev_used = set() 
	for (i, candidate_list) in enumerate(path_dag):
		curr_layer = []	
		layer_score_list = []
		used = set()  # Keep track of used vertices to avoid choosing bad edges.
		candidate_list.sort(key=lambda x: x[1], reverse=True)
		for (j, ((v, w), s)) in enumerate(candidate_list):
			if np.random.random() <= R_SKIP_TH:
				continue
			if v in used or w in used:
				# Vertex is occupied.
				continue
			if i > 0 and not (v in prev_used or w in prev_used): 
				# This edge is a straggling edge.
				continue
			# If we are at this point, we will choose the highest scoring edge.
			curr_layer.append((v, w))
			soln_size += 1
			used.add(v)
			used.add(w)
			layer_score_list.append(j)
		soln_layers.append(curr_layer)
		swap_score_list.append(layer_score_list)
		prev_used = used
	return soln_layers, soln_size, swap_score_list

def fold(candidate, coupling_map, current_layout, next_target_set):	
	# Choose a fold such that we minimize the distance to the next targets.
	min_d_index = 0
	min_dist = -1
	for i in range(len(candidate)):  # Compute point of minimum distance -- that will be the fold point.
		test_layout = current_layout.copy()
		# Perform reverse swaps
		j = 0
		while j < i:
			layer = candidate[j]
			for (p0, p1) in layer:
				test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
			j += 1
		j = len(candidate) - 1
		while j >= i:
			layer = candidate[j]
			for (p0, p1) in layer:
				test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
			j -= 1
		# After test layout is completed, compute cumulative distance.
		dist = sum(coupling_map.distance_matrix[test_layout[v0], test_layout[v1]]\
						for (v0, v1) in next_target_set)
		if min_dist < 0 or dist < min_dist:
			min_dist = dist
			min_d_index = i
	# Once we compute the min_d_index, we fold the candidate.
	if min_d_index == 0:
		return candidate[::-1][:-1], min_dist  # Return reverse of current candidate.
	elif min_d_index == len(candidate) - 1:
		return candidate[:-1], min_dist  # This is just the current order.
	else:
		left_tree = candidate[:min_d_index]
		right_tree = candidate[min_d_index+1:][::-1]
		folded_tree = []
		# We can be lazy and not fold :) Just do [left_tree, right_tree]
		folded_tree.extend(left_tree)
		folded_tree.extend(right_tree)
		return folded_tree, min_dist

