"""
    author: Suhas Vittal
    date:   29 September 2021 @ 2:09 p.m. EST
"""

from qiskit.circuit import Qubit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from fs_layerview import LayerViewPass
from fs_selector import ForeSightSelector
from fs_sum_tree import SumTreeNode
from fs_dist import process_coupling_map

import numpy as np

from copy import copy, deepcopy
from collections import deque

class ForeSight(TransformationPass):
    def __init__(self,
        coupling_map, 
        slack=2,
        solution_cap=32,
        edge_weights=None,
        vertex_weights=None,
        readout_weights=None,
        depth_minimize=False,
        debug=False
    ):
        super().__init__()

        self.slack = slack
        self.coupling_map = coupling_map        
        self.solution_cap = solution_cap
        self.depth_min = depth_minimize

        self.distance_matrix, self.paths_on_arch = process_coupling_map(coupling_map, slack, edge_weights=edge_weights) 
        self.mean_degree = len(self.coupling_map.get_edges()) / self.coupling_map.size()

        self.fake_run = False
        self.debug = debug

        # initialize weights
        self.edge_weights = edge_weights
        self.vertex_weights = vertex_weights
        self.readout_weights = readout_weights
            
    def run(self, dag):
        mapped_dag = dag._copy_circuit_metadata()
        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        
        primary_layer_view, secondary_layer_view =\
            self.property_set['primary_layer_view'], self.property_set['secondary_layer_view']
        if primary_layer_view is None or secondary_layer_view is None:
            layer_view_pass = LayerViewPass();
            layer_view_pass.run(dag)
            primary_layer_view, secondary_layer_view =\
                layer_view_pass.property_set['primary_layer_view'], layer_view_pass.property_set['secondary_layer_view']
        # Convert to deque's for lower latency
        primary_layer_view = deque(primary_layer_view)
        secondary_layer_view = deque(secondary_layer_view)

        solutions = self.deep_solve(
            primary_layer_view, 
            secondary_layer_view, 
            [(current_layout, [], 0)], 
            canonical_register 
        )

        # Choose solution with minimum depth.
        min_solution = None
        min_layout = None
        min_size = -1
        min_depth = -1
        for (layout, soln, size) in solutions:
            depth = len(soln)
            if min_solution is None or (size < min_size) or (size == min_size and depth < min_depth):
                min_layout = layout
                min_solution = soln
                min_size = size
                min_depth = depth
        self.property_set['final_layout'] = min_layout

        if self.fake_run:
            return mapped_dag   
        # Else build the dag.
        for layer in min_solution:
            for node in layer:
                mapped_dag.apply_operation_back(op=node.op, qargs=node.qargs, cargs=node.cargs)
        # validate mapped dag (SWAPs already validated -- ops are valid, just check if all ops are there)
        orig_ops = dag.count_ops()
        mapped_ops = mapped_dag.count_ops()
        for g in orig_ops:
            if orig_ops[g] != mapped_ops[g]:
                print('Error! Unequal non-SWAP gates after routing.')
                print('Original circuit: ', orig_ops)
                print('Mapped circuit: ', mapped_ops)
        return mapped_dag
    
    def deep_solve(
        self, 
        primary_layer_view, 
        secondary_layer_view, 
        base_solver_queue, 
        canonical_register 
    ):
        if len(primary_layer_view) == 0:
            return base_solver_queue

        # Build tree of output layers.
        solver_queue = []
        leaves = []
        for (i, (base_layout, base_output_layers, base_sum)) in enumerate(base_solver_queue):
            root_node = SumTreeNode(base_output_layers, base_sum, None, [])  
            leaves.append(root_node)
            solver_queue.append((base_layout, i))
        look = 0
        while len(solver_queue) <= 2*self.solution_cap:
            if self.debug:
                print('Layer Number:', len(primary_layer_view))
                print('\tSolver Queue Size:', len(solver_queue))
            if len(primary_layer_view) == 0:
                break
            next_solver_queue = []
            next_leaves = []
            for (current_layout, parent_id) in solver_queue:  # Empty out queue into next_solver_queue.
                parent = leaves[parent_id]
                # Find parent corresponding to parent id.
                solutions = self.shallow_solve(
                    primary_layer_view,
                    secondary_layer_view,
                    current_layout,
                    canonical_register
                )
                # Apply solutions non-deterministically to current_dag.
                for (i, (output_layers, new_layout, num_swaps)) in enumerate(solutions):
                    # Create node for each candidate solution.
                    node = SumTreeNode(output_layers, parent.sum_data + num_swaps, parent, []) 
                    parent.children.append(node)
                    next_leaves.append(node)
                    next_solver_queue.append((new_layout, len(next_leaves) - 1))
            solver_queue = next_solver_queue
            leaves = next_leaves
            primary_layer_view.popleft()
            secondary_layer_view.popleft()
            look += 1
        # Now, we simply check the leaves of the output layer tree. We select the leaf with the minimum sum.
        min_leaves = []
        min_sum = -1
        for (i, leaf) in enumerate(leaves):
            layout, _ = solver_queue[i]
            if min_sum < 0 or leaf.sum_data < min_sum:
                min_leaves = [(leaf, layout, leaf.sum_data)]
                min_sum = leaf.sum_data
            elif leaf.sum_data == min_sum:
                min_leaves.append((leaf, layout, leaf.sum_data))
        # Now that we know the best leaves, we simply just need to build the corresponding dags.
        # Traverse up the leaves -- there is only one path to a root.
        if len(min_leaves) > self.solution_cap:
            min_leaf_indices = np.random.choice(np.arange(len(min_leaves)), size=self.solution_cap)
            min_leaves = [min_leaves[i] for i in min_leaf_indices]
        min_solutions = []
        for (leaf, layout, leaf_sum) in min_leaves:
            output_layer_deque = deque([])
            curr = leaf
            while curr != None:
                output_layer_deque.extendleft(list(curr.obj_data)[::-1])
                curr = curr.parent
            min_solutions.append((layout, output_layer_deque, leaf_sum))
        return self.deep_solve(primary_layer_view, secondary_layer_view, min_solutions, canonical_register)
                
    def shallow_solve(
        self, 
        primary_layer_view,
        secondary_layer_view,
        current_layout,
        canonical_register
    ):
        output_layers = [[]]
        starting_output_layer = 1
        
        target_list = []
        path_collection_list = []
        post_ops = []
        target_to_op = {}
        # Only execute operations in current layer.
        # Define post primary layer view
        if len(primary_layer_view) == 1:
            post_primary_layer_view = []
        else:
            post_primary_layer_view = []
            visited = set()
            for i in range(1, min(len(primary_layer_view), int(np.ceil(10*self.mean_degree)))):
                curr_layer = []
                for node in primary_layer_view[i]:
                    q0, q1 = node.qargs
                    if q0 in visited or q1 in visited:
                        visited.add(q0)
                        visited.add(q1)
                        continue
                    visited.add(q0)
                    visited.add(q1)
                    curr_layer.append(node)
                if primary_layer_view[i]:
                    post_primary_layer_view.append(curr_layer)

        # Process operations in layer
        for op in secondary_layer_view[0]:
            output_layers[0].append(self._remap_gate_for_layout(op, current_layout, canonical_register))
        front_layer = primary_layer_view[0]
        for op in front_layer:  
            q0, q1 = op.qargs
            # Filter out all operations that are currently adjacent.
            if self.coupling_map.graph.has_edge(current_layout[q0], current_layout[q1]):
                output_layers[0].append(self._remap_gate_for_layout(op, current_layout, canonical_register))
            else:  # Otherwise, get path candidates and place it in path_collection_list.
                path_collection_list.append(self.path_find_and_fold(q0, q1, post_primary_layer_view, current_layout))
                post_ops.append(op)
                target_list.append((q0, q1))
                target_to_op[(q0, q1)] = op

        # Build PPC and get candidate list.
        if len(path_collection_list) == 0:
            return [(output_layers, current_layout, 0)] 
        path_selector = ForeSightSelector(path_collection_list, len(self.coupling_map.physical_qubits), len(path_collection_list))
        candidate_list, suggestions = path_selector.find_and_join(self, target_list, current_layout, post_primary_layer_view)
        if candidate_list is None:  # We failed, take the suggestions.
            
            tmp_pl_view = deepcopy(primary_layer_view)
            tmp_sl_view = deepcopy(secondary_layer_view)
            # Remove top layer from both views.
            tmp_pl_view.popleft()
            tmp_sl_view.popleft()
            # Idea: run shallow solve on suggestions, perform cross product on results.
            target_lists = []
            for index_list in suggestions:
                if len(index_list) == 0:
                    continue
                target_sub_list = [target_list[i] for i in index_list]
                target_lists.append(target_sub_list)
                tmp_pl_view.appendleft([target_to_op[target] for target in target_sub_list])
                tmp_sl_view.appendleft([])
            solutions = [(output_layers, current_layout, 0)]
            while target_lists:
                target_sub_list = target_lists.pop()
                next_solutions = []
                for (prev_layers, prev_layout, prev_swaps) in solutions:
                    solution_list = self.shallow_solve(
                        tmp_pl_view,
                        tmp_sl_view,
                        prev_layout,
                        canonical_register
                    )
                    for (layers, layout, num_swaps) in solution_list:
                        prev_layers_cpy = copy(prev_layers)
                        prev_layers_cpy.extend(layers)
                        next_solutions.append((prev_layers_cpy, layout, prev_swaps+num_swaps))
                tmp_pl_view.popleft()
                tmp_sl_view.popleft()

                next_solution_cap = int(np.ceil(np.log2(self.solution_cap) + 1))
                if len(next_solutions) > 2*next_solution_cap:
                    kept_soln_indices = np.random.choice(np.arange(len(next_solutions)), size=next_solution_cap)  
                    next_solutions = [next_solutions[k] for k in kept_soln_indices]
                solutions = next_solutions
            return solutions

        # Compute all solutions.
        solutions = []
        
        for soln in candidate_list:
            output_layers_cpy = deepcopy(output_layers)
            new_layout = current_layout.copy()
            num_swaps = 0
            for (i, layer) in enumerate(soln):  # Perform the requisite swaps.
                output_layers_cpy.append([])
                for (p0, p1) in layer:
                    if p0 == p1:
                        continue
                    v0, v1 = new_layout[p0], new_layout[p1]
                    swp_gate = DAGNode(
                        type='op',
                        op=SwapGate(),
                        qargs=[v0, v1]  
                    )
                    output_layers_cpy[-1].append(self._remap_gate_for_layout(swp_gate, new_layout, canonical_register))
                    # Apply swap to modify running layout.
                    new_layout.swap(p0, p1)
                    num_swaps += 1
            # Apply the operations after completing all the swaps.
            output_layers_cpy.append([])
            for op in post_ops:  
                q0, q1 = op.qargs
                if not self.coupling_map.graph.has_edge(new_layout[q0], new_layout[q1]):
                    print('ERROR: not satisified %d(%d), %d(%d)'\
                        % (q0.index, current_layout[q0], q1.index, current_layout[q1]))
                    print('%d edges:' % current_layout[q0], self.coupling_map.neighbors(current_layout[q0]))
                    print('%d edges:' % current_layout[q1], self.coupling_map.neighbors(current_layout[q1]))
                    print('|TargetSet| = %d' % len(target_list))
                    print('Solution: ', soln)
                    exit()
                output_layers_cpy[-1].append(self._remap_gate_for_layout(op, new_layout, canonical_register))
            solutions.append((output_layers_cpy, new_layout, num_swaps))  # Save layers to apply to DAG and corresponding layout.
        return solutions

    def path_find_and_fold(self, v0, v1, post_primary_layer_view, current_layout):
        p0, p1 = current_layout[v0], current_layout[v1]
        if self.coupling_map.graph.has_edge(p0, p1):
            return []
        # Get path candidates from _path_select.
        path_candidates = self.paths_on_arch[(p0, p1)]
        path_folds = []
        for path in path_candidates:
            path_folds.extend(self._path_minfold(path, current_layout, post_primary_layer_view))
        return path_folds
    
    def _path_minfold(self, path, current_layout, post_primary_layer_view):
        min_dist = -1
        min_folds = []
        latest_layout = current_layout
        latest_fold = None
        for i in range(len(path)):
            fold = [] if latest_fold is None else copy(latest_fold)
            test_layout = latest_layout.copy()
            # Perform all swaps except path[i].
            # If i = 0, then perform fold that eliminates first swap.
            if i == 0:
                j = len(path) - 1
                while j > i:
                    p0, p1 = path[j]
                    fold.append(path[j])
                    test_layout[p0], test_layout[p1] = test_layout[p1], test_layout[p0]
                    j -= 1
            else:  # if i != 0, then undo last backwards swap in last fold, and perform forwards swap prior to folded swap. 
                f0, f1 = path[i-1]
                b0, b1 = path[i]
                test_layout[b0], test_layout[b1] = test_layout[b1], test_layout[b0]  # Flip them back.
                test_layout[f0], test_layout[f1] = test_layout[f1], test_layout[f0]  # Perform forward flip.
                fold.pop()  # Remove last swap.
                fold.insert(i-1, path[i-1])  # Insert new swap at beginning.
            # Update latest layout.
            latest_layout = test_layout
            latest_fold = fold
            # Compute distance to post primary layer.
            dist = self._distf(len(path)-1, post_primary_layer_view, test_layout)
            if dist < min_dist or min_dist == -1:
                min_dist = dist
                min_folds = [(
                    fold, 
                    dist
                )]
            elif dist == min_dist:
                min_folds.append((
                    fold, 
                    dist
                ))
        return min_folds

    def _remap_gate_for_layout(self, op, layout, canonical_register):
        new_op = copy(op)
        new_op.qargs = [canonical_register[layout[x]] for x in op.qargs]
        return new_op

    def _verify_and_measure(
        self, 
        soln,
        target_list,
        current_layout,
        post_primary_layer_view,
        verify_only=False
    ):
        if len(target_list) == 0:
            return True, 0
        test_layout = current_layout.copy()
        size = 0
        for layer in soln:
            for (p0, p1) in layer:
                test_layout.swap(p0, p1)
                size += 1
        max_allowed_size = 0
        for (v0, v1) in target_list:
            max_allowed_size += self.distance_matrix[current_layout[v0]][current_layout[v1]]
            p0, p1 = test_layout[v0], test_layout[v1]
            if not self.coupling_map.graph.has_edge(p0, p1):
                return False, 0
        # If we have gotten to this point, then our soln is good.
        if verify_only:
            dist = 0
        else:
            dist = self._distf(size, post_primary_layer_view, test_layout)

        return True, dist

    def _distf(self, soln_size, post_primary_layer_view, test_layout):
        dist = 0.0
        num_ops = 0
        for r in range(0, len(post_primary_layer_view)):
            post_layer = post_primary_layer_view[r]
            sub_sum = 0.0
            for node in post_layer:
                q0, q1 = node.qargs
                p0, p1 = test_layout[q0], test_layout[q1]
                num_ops += 1
                s = self.distance_matrix[p0][p1]
                p = 0
                if self.vertex_weights:
                    # Modify with vertex weights
                    p += 2*(self.vertex_weights[p0]+self.vertex_weights[p1])
                if self.readout_weights:
                    # Modify with readout weights
                    max_post_layer_size = np.ceil(10*self.mean_degree)
                    p += np.exp(10*(self.readout_weights[p0]+self.readout_weights[p1])
                        *(max_post_layer_size-len(post_primary_layer_view))/max_post_layer_size)
                sub_sum += s+0.5*p
            if self.depth_min:  
                dist += sub_sum
            else:
                dist += sub_sum * np.exp(-(r/((self.mean_degree)**2))**2)
        if num_ops == 0:
            return 0
        else:
            dist = dist/num_ops
            soln_size = soln_size*np.min(self.distance_matrix)
            if self.depth_min:
                return dist + soln_size
            else:
                return dist+soln_size*np.exp(-(num_ops/self.mean_degree)**2)  # scale down

