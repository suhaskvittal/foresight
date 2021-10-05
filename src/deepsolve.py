"""
	author: Suhas Vittal
	date: 29 September 2021 @ 2:09 p.m EST
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

import numpy as np

from copy import copy, deepcopy

from solnsearch import compute_candidate, fold
from pathfind import build_path_dag, _path_find

def deep_solve(coupling_map, dag, dag_in_progress, in_layout, in_layer, finished_ops, lookahead=5, max_swaps=10):
	if len(in_layer) == 0:
		return (in_layer, in_layout, dag_in_progress, finished_ops)

	front_layer_queue = [(in_layer, in_layout, dag_in_progress, finished_ops)]
	# Essentially a BFS on the possible layouts.
	for _ in range(lookahead):
		next_front_layer_queue = []
		while len(front_layer_queue) > 0:
			front_layer, current_layout, curr_dag, finished = front_layer_queue.pop(0)
			if len(front_layer) == 0:
				next_front_layer_queue.append((front_layer, current_layout, mapped_dag, br_finished))
			second_layer_set = set()
			for node in front_layer:
				for child in dag.descendants(node):
					second_layer_set.add(child)
			second_layer = list(second_layer_set)

			solutions = shallow_solve(coupling_map, dag, current_layout, front_layer, second_layer, max_swaps)
			for (output_layers, next_layout) in solutions:
				mapped_dag = deepcopy(curr_dag)
				br_finished = copy(finished)
				for layer in output_layers:
					for node in layer:
						mapped_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
				for node in front_layer:
					br_finished.add(node)
				next_layer = []
				for node in front_layer:
					adj_list = dag.descendants(node)
					for next_node in adj_list:
						if all(x in br_finished for x in dag.ancestors(next_node) if x.type == 'op')\
						and next_node not in finished and next_node not in next_layer:
							next_layer.append(next_node)
				next_front_layer_queue.append((next_layer, next_layout, mapped_dag, br_finished))
		front_layer_queue.extend(next_front_layer_queue)
	# Compute dag with the minimum size
	candidates = front_layer_queue
	while len(candidates) > 1:
		# Compute min candidates
		min_size = -1
		min_outcomes = []
		for entry in candidates:
			_, _, mapped_dag, _ = entry
			if min_size == -1 or mapped_dag.size() < min_size:
				min_size = mapped_dag.size()
				min_outcomes = [entry]
			elif mapped_dag.size() == min_size:
				min_outcomes.append(entry)
		# No need to continue deep solve.
		if len(min_outcomes) == 1:
			return min_outcomes[0]
		# Otherwise, continue with min candidates and run deep solve again.
		all_done = True
		next_candidates = [] 
		for opt_entry in min_outcomes:
			front_layer, current_layout, mapped_dag, finished = opt_entry
			if len(front_layer) == 0:
				next_candidates.append(opt_entry)
			else:
				all_done = False
				next_candidates.append(deep_solve(
					coupling_map,
					dag,
					mapped_dag,
					current_layout,
					front_layer,
					finished,
					lookahead=lookahead,
					max_swaps=max_swaps
				))
		candidates = next_candidates
		if all_done:
			# If all_done = True, then we cannot traverse deeper. Choose a random solution.
			return min_outcomes[np.random.randint(0, high=len(min_outcomes))]
	return candidates[0]

def shallow_solve(coupling_map, dag, current_layout, front_layer, next_layer, max_swaps):
	output_layers = [[] for _ in range(max_swaps+2)]

	canonical_register = dag.qregs['q']
	
	target_set = []			# Current set of two-qubit ops			
	next_target_set = []	# Next set of two-qubit ops
	post_ops = []
	for gnode in front_layer:
		if gnode.type != 'op':
			continue
		if len(gnode.qargs) != 2 or gnode.name == 'measure':
			output_layers[0].append(_remap_gate_for_layout(gnode, current_layout, canonical_register))
		else:
			v0, v1 = gnode.qargs
			if coupling_map.graph.has_edge(current_layout[v0], current_layout[v1]):
				output_layers[0].append(_remap_gate_for_layout(gnode, current_layout, canonical_register))
			else:
				target_set.append((v0, v1))			
				post_ops.append(gnode)
	for gnode in next_layer:
		if gnode.type != 'op' or gnode.name == 'measure' or len(gnode.qargs) != 2:
			continue  # Do not update output layer
		else:
			v0, v1 = gnode.qargs
			next_target_set.append((v0, v1))
	if len(target_set) == 0:  # No swaps required.
		return [([output_layers[0]], current_layout)]
	routing_candidates = _find_routing_candidates(coupling_map, current_layout, max_swaps, target_set, next_target_set)	
	# Simulate each routing candidate.
	solutions = []
	for candidate_soln in routing_candidates:
		output_layers_cpy = deepcopy(output_layers)  # Make a copy of the output layers
		new_layout = current_layout.copy()
		for (i, layer) in enumerate(candidate_soln):  # Perform the requisite swaps.
			for (p0, p1) in layer:
				v0, v1 = new_layout[p0], new_layout[p1]
				if (v0, v1) in target_set or (v1, v0) in target_set:
					continue
				swp_gate = DAGNode(
					type='op',
					op=SwapGate(),
					qargs=[v0, v1]	
				)
				output_layers_cpy[i+1].append(_remap_gate_for_layout(swp_gate, new_layout, canonical_register))
				# Apply swap to modify running layout.
				new_layout[p0], new_layout[p1] = new_layout[p1], new_layout[p0]
		for gnode in post_ops:  # Apply original operationsafter swapping.
			output_layers_cpy[i+2].append(_remap_gate_for_layout(gnode, new_layout, canonical_register))
		# Add output_layers_cpy and final layout to solutions
		solutions.append((output_layers_cpy, new_layout))
	return solutions

def _remap_gate_for_layout(gnode, layout, canonical_register):
	new_gnode = copy(gnode)
	new_gnode.qargs = [canonical_register[layout[x]] for x in gnode.qargs]
	return new_gnode

def _find_routing_candidates(coupling_map, current_layout, max_swaps, target_set, next_target_set, rs_steps=1000):
	path_dag = build_path_dag(coupling_map, target_set, current_layout, max_swaps)
	# Compute routing candidates.
	routing_candidates = []
	routing_hash_store = set()  # Used to keep track of identified candidates

	best_soln_size = -1
	best_next_dist = -1
	for _ in range(rs_steps):	
		candidate_soln, soln_size, swap_score_list = compute_candidate(path_dag)
		candidate_soln, next_dist = fold(candidate_soln, coupling_map, current_layout,\
													next_target_set=next_target_set)
		score_modify = 1.0
		candidate_hash = _soln_hash_f(candidate_soln)
		if candidate_hash in routing_hash_store:
			continue  # do not add candidate
		if _verify_swaps(candidate_soln, coupling_map, target_set, current_layout):
			routing_hash_store.add(candidate_hash)
			#if soln_size < best_soln_size or best_soln_size == -1:
			if (soln_size + next_dist < best_soln_size + best_next_dist) or best_soln_size == -1:
				routing_candidates = [candidate_soln]
				best_soln_size = soln_size
				best_next_dist = next_dist
			#elif soln_size == best_soln_size:
			elif soln_size + next_dist == best_soln_size + best_next_dist:
				routing_candidates.append(candidate_soln)
			else:
				score_modify = 0.5
		else:
			score_modify = 0.5
		# Update scores for candidate solution
		for (i, score_layer) in enumerate(swap_score_list):
			for j in score_layer:
				(v, w), s = path_dag[i][j]
				path_dag[i][j] = (v, w), (score_modify*np.random.random()+score_modify)*s
	return routing_candidates

def _verify_swaps(candidate_soln, coupling_map, target_set, current_layout):
	test_layout = current_layout.copy()
	target_set_indicator = {x: 0 for x in target_set}
	for layer in candidate_soln:
		for (p0, p1) in layer:
			test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
	for p0 in coupling_map.physical_qubits:
		v0 = test_layout[p0]
		for p1 in coupling_map.neighbors(p0):
			v1 = test_layout[p1]
			if (v0, v1) in target_set:
				target_set_indicator[(v0, v1)] = 1
			elif (v1, v0) in target_set:
				target_set_indicator[(v1, v0)] = 1
	return all(target_set_indicator[x] == 1 for x in target_set)
			
def _soln_hash_f(soln):
	h = 0
	PRIME = 766453 
	for layer in soln:
		for (p0, p1) in layer:
			h += ((2**p0)*(3**p1)) % PRIME
	return h	
