"""
    author: Suhas Vittal
    date:   29 September 2021
"""

from qiskit.circuit import Qubit

from qiskit.compiler import transpile
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import *
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from fs_util import *
from fs_util import _path_to_swap_collection

from foresight.fs_dist import process_coupling_map
from foresight.fs_multipath import SolutionKernel
from foresight.fs_path_join_tree import PathJoinTree
from foresight.fs_path_priority_queue import PathPriorityQueue
from foresight.fs_hashed_layout import HashedLayout

import numpy as np

from copy import copy, deepcopy
from collections import deque, defaultdict

FLAG_DEBUG = 0x1
FLAG_ALAP = 0x2
FLAG_ASAP = 0x4
FLAG_NOISE_AWARE = 0x8
FLAG_OPT_FOR_O3 = 0x10

DEFAULT_FLAGS = FLAG_ALAP

def cmp(score1, score2, size1, size2):  # Returns true if 1 < 2
    return (score1 < score2) or (score1 == score2 and size1 < size2)

class ForeSight(TransformationPass):
    def __init__(self,
        coupling_map, 
        slack=2,
        solution_cap=32,
        cx_error_rates=None,
        sq_error_rates=None,
        ro_error_rates=None,
        flags=DEFAULT_FLAGS
    ):
        """
        Initializes ForeSight pass.

        coupling_map: backend coupling graph
        slack: expands path consideration. If slack=0, then shortest paths are
            considered. If slack=1, then shortest and second shortest paths are 
            considered. (etc.)
        solution_cap: caps size of the computation tree. Tree is contracted
            if treewidth exceeds 2*solution_cap.
        """
        super().__init__()

        self.slack = slack
        self.coupling_map = coupling_map        
        self.solution_cap = solution_cap

        # Runtime data structures
        self.input_dag = None
        self.canonical_register = None
        self.solutions = []

        # Other
        self.fake_run = False
        self.debug = flags & FLAG_DEBUG
        self.alap_enabled = flags & FLAG_ALAP
        self.asap_enabled = flags & FLAG_ASAP
        self.noise_aware = flags & FLAG_NOISE_AWARE
        self.using_o3 = flags & FLAG_OPT_FOR_O3

        # initialize weights
        self.cx_error_rates = cx_error_rates
        self.sq_error_rates = sq_error_rates
        self.ro_error_rates = ro_error_rates
        
        self.distance_matrix, self.paths_on_arch = process_coupling_map(
            coupling_map,
            slack,
            edge_weights=cx_error_rates if self.noise_aware else None,
        ) 
        self.mean_degree = len(self.coupling_map.get_edges()) / self.coupling_map.size()

        # usage statistics
        self.suggestions = 0
        self.iterations = 0
        self.prunings = 0
        self.swap_segments = []
            
    def run(self, dag):
        """
            Compiler pass main method. 
            We do the following:
                (1) Compute the primary layer view and secondary
                    layer view.
                (2) Run deep solve on the given dag.
                (3) Given the outputs from deep solve, we evaluate
                    each solution by observing the output size
                    and depth. We prioritize lower size over lower
                    depth.
                (4) We then finally route based on the proposed
                    solution and return the result.
            ForeSight also verifies the result during execution
                We verify in two places.
                    (a) Shallow Solve -- we verify that all
                        2-qubit operations in the layer have
                        adjacent operands.
                    (b) In this method -- we verify that all
                        operations in the original dag have
                        been placed.
        """
        mapped_dag = dag._copy_circuit_metadata()

        # Clear global data structures
        self.suggestions = 0
        self.iterations = 0
        self.prunings = 0
        self.swap_segments.clear()
        self.solutions.clear()
        # Initialize global data structures
        self.input_dag = dag
        self.canonical_register = dag.qregs["q"]
        # Compute initial data structures
        front_layer = []
        initial_pred_table = defaultdict(int)
        initial_layout = Layout.generate_trivial_layout(self.canonical_register)
        for (_, input_node) in dag.input_map.items():
            for s in self._successors(input_node):
                initial_pred_table[s] += 1
                if initial_pred_table[s] == len(s.qargs):
                    front_layer.append(s)
        completed_nodes = set()
        next_layer = self.compute_next_layer(front_layer, initial_pred_table, completed_nodes) 
        init_kernel = SolutionKernel(
            front_layer,
            next_layer,
            initial_pred_table,
            completed_nodes,
            deque([]),          # Schedule
            initial_layout, 
            0,                  # Swap count
            1.0,                # Expected prob success
            None                # Parent solution kernel
        )
        self.solutions = [init_kernel]
        # Start execution
        completed_solutions = []
        cycle = 0
        prunings = 0
        while len(self.solutions) > 0:
            next_solutions = []
            if self.debug and cycle % 50 == 0:
                print('suggestions:', self.suggestions)
                print('number of solutions:', len(self.solutions))
                print('number of completed solutions', len(completed_solutions))
            for i in range(len(self.solutions)):
                kernel = self.solutions[i]
                if self.debug and cycle % 50 == 0 and i == 0:
                    print('kernel %d' % i)
                    print('\tswaps used', kernel.swap_count)
                    print('\tgates completed:', len(kernel.completed_nodes))
                    print('\tfront layer:')
                    for node in kernel.front_layer:
                        print('\t\t%s' % node.name, end=' ')
                        for q in node.qargs:
                            print('%d(%d)' % (q.index, kernel.layout[q]), end=' ')
                        print('\n', end='')
                if len(kernel.front_layer) == 0:
                    curr = kernel.parent
                    while curr is not None:
                        i = len(curr.schedule)-1
                        while i >= 0:
                            kernel.schedule.appendleft(curr.schedule[i])
                            i -= 1
                        curr = curr.parent
                    kernel.parent = None
                    completed_solutions.append(kernel)
                    continue
                children = self.explore_kernel(kernel)
                next_solutions.extend(children)
            self.solutions = next_solutions
            if len(self.solutions) > self.solution_cap:
                self.solutions = self.contract_solutions(self.solutions)
                prunings += 1
            cycle += 1
        self.iterations = cycle
        self.prunings = prunings

        completed_solutions = self.contract_solutions(completed_solutions)
        # Choose solution with minimum swaps.
        if self.using_o3:
            mapped_dag = None
            # Run O3 on all completed solutions, choose the one with the minimum cnot count. 
            for (i, kernel) in enumerate(completed_solutions): 
                test_dag = dag._copy_circuit_metadata()
                for layer in kernel.schedule:
                    for node in layer:
                        test_dag.apply_operation_back(
                                op=node.op, qargs=node.qargs, cargs=node.cargs)
                # Validate test dag.
                orig_ops = dag.count_ops()
                mapped_ops = test_dag.count_ops()
                # Self-verify that no operations have been skipped.
                error_found = False
                for g in orig_ops:
                    if g not in mapped_ops or orig_ops[g] != mapped_ops[g]:
                        print('Error! Unequal non-SWAP gates after routing.')
                        print('\tOriginal circuit: ', orig_ops)
                        print('\tMapped circuit: ', mapped_ops)
                        error_found = True
                        break
                if error_found:
                    continue
                # Apply Qiskit O3 to the test dag
                test_circ = dag_to_circuit(test_dag)
                try:
                    test_circ = transpile(
                        test_circ,
                        coupling_map=self.coupling_map,
                        basis_gates=G_QISKIT_GATE_SET,
                        layout_method='trivial',
                        routing_method='none',
                        optimization_level=3
                    )
                except:
                    test_circ = transpile(
                        test_circ,
                        coupling_map=self.coupling_map,
                        basis_gates=G_QISKIT_GATE_SET,
                        layout_method='trivial',
                        routing_method='none',
                        optimization_level=0
                    )
                test_dag = circuit_to_dag(test_circ)
                cnots = test_dag.count_ops()['cx']
                depth = test_dag.depth()
                if self.debug:
                    print('kernel %d of %d has %d cnots and %d depth (originally %d swaps)'\
                            % (i, len(completed_solutions), cnots, depth, kernel.swap_count)) 
                if mapped_dag is None:
                    mapped_dag = test_dag
                else:
                    min_cnots = mapped_dag.count_ops()['cx']
                    min_depth = mapped_dag.depth()
                    if cmp(cnots, min_cnots, depth, min_depth):
                        mapped_dag = test_dag
        else:
            if self.noise_aware:
                best_solution = max(completed_solutions, key=lambda x: x.expected_prob_success)
            else:
                best_solution = min(completed_solutions, key=lambda x: x.swap_count)
            self.swap_segments = best_solution.swap_segments
            self.property_set['final_layout'] = best_solution.layout
            if self.fake_run:
                return mapped_dag   
            # Else build the dag.
            for layer in best_solution.schedule:
                for node in layer:
                    mapped_dag.apply_operation_back(op=node.op, qargs=node.qargs, cargs=node.cargs)
            # Validate mapped dag
            # SWAPs already validated -- ops are valid, just check if all ops are there
            orig_ops = dag.count_ops()
            mapped_ops = mapped_dag.count_ops()
            # Self-verify that no operations have been skipped.
            for g in orig_ops:
                if g not in mapped_ops or orig_ops[g] != mapped_ops[g]:
                    print('Error! Unequal non-SWAP gates after routing.')
                    print('\tOriginal circuit: ', orig_ops)
                    print('\tMapped circuit: ', mapped_ops)
                    break
            if self.debug:
                print('Number of swaps: %d' % best_solution.swap_count)
                print('EPS: %f' % best_solution.expected_prob_success)
        if self.debug:
            print('Statistics')
            print('\tSuggestion usage:', self.suggestions)
            print('\tPrunings to total iterations: %d to %d' % (self.prunings, self.iterations))
        return mapped_dag

    def contract_solutions(self, solutions, prune_cap=None):
        layout_table = {}
        for kernel in solutions:
            layout = kernel.layout
            hashed_layout = HashedLayout.from_layout(layout)

            if self.noise_aware:
                score = (1.0-kernel.expected_prob_success) / len(kernel.completed_nodes)
            else:
                score = kernel.swap_count / len(kernel.completed_nodes)
            if hashed_layout not in layout_table:
                layout_table[hashed_layout] = (kernel, score)
            else:
                min_kernel, min_score = layout_table[hashed_layout]
                if cmp(score, min_score, kernel.swap_count, min_kernel.swap_count):
                    layout_table[hashed_layout] = (kernel, score)
        min_kernels = [layout_table[hl] for hl in layout_table]
        if prune_cap is None:
            prune_cap = self.solution_cap // 2
        prune_cap = max(prune_cap, 1)
        if len(min_kernels) > prune_cap:
            min_kernels.sort(key=lambda x: x[1])
            _, min_score = min_kernels[0]
            filtered_kernels = []
            for i in range(prune_cap):
                kernel, score = min_kernels[i]
                if min_score <= score <= min_score*1.5:
                    filtered_kernels.append((kernel,score))
            min_kernels = filtered_kernels
        solutions = []
        for (kernel, _) in min_kernels:
            curr = kernel.parent
            while curr is not None:
                i = len(curr.schedule)-1
                while i >= 0:
                    kernel.schedule.appendleft(curr.schedule[i])
                    i -= 1
                curr = curr.parent
            kernel.parent = None
            solutions.append(kernel)
        return solutions

    def explore_kernel(self, kernel):
        front_layer = kernel.front_layer.copy()
        next_layer = kernel.next_layer.copy()
        pred_table = kernel.pred_table.copy()
        completed_nodes = kernel.completed_nodes.copy()
        current_layout = kernel.layout.copy()

        schedule = deque([])

        # Do-while loop, but in python :p
        exec_list = []
        while True:  # do ... while exec_list is nonempty
            schedule.append([])
            exec_list.clear()
            # Initialize data structures
            next_front_layer = []
            for node in front_layer:
                if node in completed_nodes:
                    exec_list.append(node)
                elif len(node.qargs) != 2:
                    exec_list.append(node)
                else:
                    q0, q1 = node.qargs
                    if self.coupling_map.graph.has_edge(current_layout[q0], current_layout[q1]):
                        exec_list.append(node)
                    else:
                        next_front_layer.append(node)
            # Complete gates in the exec_list that are not already completed.
            visited = set()  # maintain a visited set to avoid double counting any gates
            for node in front_layer:
                visited.add(node)
            for node in exec_list:
                if node not in completed_nodes:
                    schedule[-1].append(self._remap_gate_for_layout(node, current_layout))
                for s in self._successors(node):
                    if s in visited:
                        continue
                    if node not in completed_nodes:
                        pred_table[s] += 1
                    if pred_table[s] == len(s.qargs) and self.asap_enabled:
                        next_front_layer.append(s)
                        visited.add(s)
                completed_nodes.add(node)
            # Update front layer
            if self.alap_enabled and len(next_front_layer) == 0:
                front_layer = next_layer
                next_layer = self.compute_next_layer(front_layer, pred_table, completed_nodes)
            else:
                front_layer = next_front_layer
            # Only restart loop if we had executed anything
            if len(exec_list) == 0:
                break
        # Everything currently in the front layer needs to be finished.
        mid_kernel = SolutionKernel(
            front_layer,
            next_layer,
            pred_table,
            completed_nodes,
            schedule,
            current_layout,
            kernel.swap_count,
            kernel.expected_prob_success,
            kernel,
        )
        if len(front_layer) == 0:
            candidate_list = [[]]
        elif self.alap_enabled:
            path_collection_list = []
            future_gates = self.compute_future_gates(front_layer, pred_table, completed_nodes)
            for node in front_layer:
                q0, q1 = node.qargs
                path_collection_list.append(
                        self.path_find_and_fold(mid_kernel, q0, q1, future_gates))
            candidate_list, suggestions = self.merge_solutions(
                                            mid_kernel, path_collection_list, future_gates)
            if candidate_list is None:
                self.suggestions += 1
                # Partition front layer in sublayers and explore those kernels.
                sublayers = []
                for index_list in suggestions:
                    if len(index_list) == 0:
                        continue
                    sublayer = [front_layer[i] for i in index_list]
                    sublayers.append(sublayer)
                first_subkernel = SolutionKernel(
                    [],
                    sublayers[0],
                    pred_table,
                    completed_nodes,
                    schedule,
                    current_layout,
                    kernel.swap_count,
                    kernel.expected_prob_success,
                    kernel,
                ) 
                curr_kernels = [first_subkernel]
                for i in range(len(sublayers)):
                    next_kernels = []
                    if i == len(sublayers) - 1:
                        next_sublayer = None
                    else:
                        next_sublayer = sublayers[i+1]
                    curr_sublayer = sublayers[i]
                    for subkernel in curr_kernels:
                        children = self.explore_kernel(subkernel)
                        for child in children:
                            if next_sublayer is None:
                                child.front_layer = curr_sublayer
                                child.next_layer = next_layer
                            else:
                                child.front_layer = curr_sublayer
                                child.next_layer = next_sublayer
                            next_kernels.append(child)
                    curr_kernels = next_kernels
                    # Contract if necessary
                    cap = int(np.ceil(np.log2(self.solution_cap)+1))
                    if len(curr_kernels) > cap: 
                        curr_kernels = self.contract_solutions(curr_kernels, prune_cap=cap//2)
                return curr_kernels
        else:  # asap_enabled
            path_collection_list = []
            future_gates = self.compute_future_gates(front_layer, pred_table, completed_nodes)
            for node in front_layer:
                q0, q1 = node.qargs
                path_collection_list.append(
                        self.path_find_and_fold(mid_kernel, q0, q1, future_gates))
            # Simply choose the best folds from here.
            layout_table = {}
            for (i, node) in enumerate(front_layer):
                singleton = [node]
                mid_kernel.front_layer = singleton
                for (fold, score) in path_collection_list[i]:
                    soln = _path_to_swap_collection(fold)
                    is_valid, _, layout, size = self._verify_and_measure(
                        mid_kernel,
                        soln,
                        future_gates,
                        verify_only=True
                    )
                    if not is_valid:
                        continue
                    hashed_layout = HashedLayout.from_layout(layout)
                    if hashed_layout not in layout_table:
                        layout_table[hashed_layout] = (soln, score, size)
                    else:
                        _, min_score, min_size = layout_table[hashed_layout]
                        if cmp(score, min_score, size, min_size):
                            layout_table[hashed_layout] = (soln, score, size)
            # Get candidates from layout table
            candidate_list = [layout_table[hl][0] for hl in layout_table]
        # Now, create child kernels for each solution in the candidate list.
        children = []
        for soln in candidate_list:
            # Copy data
            new_layout = current_layout.copy()
            swap_count = kernel.swap_count
            expected_prob_success = kernel.expected_prob_success
            # Copy base schedule
            new_schedule = deque([])
            # Copy swap segments
            swap_segments = kernel.swap_segments.copy()
            for layer in schedule:
                new_schedule.append([])
                for node in layer:
                    new_schedule[-1].append(node)
                        # Update EPS if we have error rates available.
                    if self.cx_error_rates is not None\
                    and self.ro_error_rates is not None\
                    and self.sq_error_rates is not None:
                        if len(node.qargs) == 1:
                            v = node.qargs[0]
                            p = v.index
                            if node.name == 'measure':
                                expected_prob_success *= 1 - self.ro_error_rates[p]
                            else:
                                expected_prob_success *= 1 - self.sq_error_rates[p]
                        elif len(node.qargs) == 2:
                            v0, v1 = node.qargs
                            p0, p1 = v0.index, v1.index
                            expected_prob_success *= 1 - self.cx_error_rates[(p0,p1)]
                        else:
                            pass
            # Add swaps now
            for layer in soln:
                new_schedule.append([])
                for (p0,p1) in layer:
                    if p0 == p1:
                        continue
                    v0, v1 = new_layout[p0], new_layout[p1]
                    swap_gate = DAGOpNode(op=SwapGate(), qargs=[v0,v1])
                    new_schedule[-1].append(self._remap_gate_for_layout(swap_gate, new_layout))
                    new_layout.swap(p0,p1)
                    swap_count += 1
                    if self.cx_error_rates is not None:
                        expected_prob_success *= (1 - self.cx_error_rates[(p0,p1)])**3
            swap_segments.append(swap_count - kernel.swap_count)
            child_kernel = SolutionKernel(
                front_layer,
                next_layer,
                pred_table,
                completed_nodes,
                new_schedule,
                new_layout,
                swap_count,
                expected_prob_success,
                kernel,
                swap_segments=swap_segments
            )
            children.append(child_kernel)
        return children
    
    def compute_future_gates(self, front_layer, pred_table, completed_nodes):
        future_gates = []
        
        incremented = []
        curr_layer = front_layer
        future_depth = np.ceil(10*self.mean_degree) 
        qubit_depth = defaultdict(int)
        if self.asap_enabled:
            i = 0
        else:
            i = -1  # First iteration is front layer, we want to ignore that
        while i < future_depth and len(curr_layer) > 0:
            next_layer = []
            had_2q_gate = False
            for node in curr_layer:
                if i >= 0 and node not in completed_nodes:
                    # Only keep gates that have both qubits unused.
                    if len(node.qargs) == 1 and self.noise_aware:
                        # Only consider such gates in noise aware mode.
                        future_gates.append((node, i))
                    elif len(node.qargs) == 2:
                        q0, q1 = node.qargs
                        depth = max(qubit_depth[q0], qubit_depth[q1])
                        qubit_depth[q0] += 1
                        qubit_depth[q1] += 1
                        future_gates.append((node,depth))
                        had_2q_gate = True
                    else:
                        pass
                for s in self._successors(node):
                    if node not in completed_nodes:
                        pred_table[s] += 1
                        incremented.append(s)
                    if pred_table[s] == len(s.qargs):
                        next_layer.append(s)
            if i < 0 or had_2q_gate:
                i += 1
            curr_layer = next_layer
        for node in incremented:
            pred_table[node] -= 1
        return future_gates

    def compute_next_layer(self, front_layer, pred_table, completed_nodes):
        incremented = []

        next_layer = []
        visited = set()
        for node in front_layer:
            for s in self._successors(node):
                if s in visited:
                    continue
                if node not in completed_nodes:
                    pred_table[s] += 1
                    incremented.append(s)
                if pred_table[s] == len(s.qargs):
                    next_layer.append(s)
                    visited.add(s)
        for node in incremented:
            pred_table[node] -= 1
        return next_layer

    def path_find_and_fold(self, kernel, v0, v1, future_gates):
        """
            We will examine all pre-computed paths between the physical qubits
            corresponding to v0 and v1. These were computed on pass instantiation
            by the Floyd-Warshall algorithm. In order to minimize depth and 
            reduce swaps, we fold paths so that the sequence of SWAPs applied 
            minimizes distance to future operations.

            v0: the source vertex
            v1: the sink vertex
            post_primary_layer_view: a BFS-list of future operations that helps
                in deciding where to fold along a path.
            current_layout: the current running layout during routing.

            Returns a list of path folds. Note that all path folds in the list are
                not necessarily equally optimal, but they are the most optimal
                for their respective path.
        """
        p0, p1 = kernel.layout[v0], kernel.layout[v1]
        if self.coupling_map.graph.has_edge(p0, p1):
            return []
        path_candidates = self.paths_on_arch[(p0, p1)]
        path_folds = []
        for path in path_candidates:
            path_folds.extend(self._path_minfold(kernel, path, future_gates))
        return path_folds

    def merge_solutions(self, kernel, path_collection_list, future_gates, runs=1000):
        front_layer = kernel.front_layer

        if len(front_layer) == 0:
            return [], None
        # If there is only one operation in the front layer,
        # then, we only need to find the best folds for that operation.
        if len(front_layer) == 1:
            prio_queue = PathPriorityQueue.buildheap(path_collection_list[0])
            layout_table = {}
            while prio_queue.size > 0:
                path, score = prio_queue.dequeue()
                soln = _path_to_swap_collection(path)
                valid, _, final_layout, size = self._verify_and_measure(
                                kernel,
                                soln,
                                future_gates,
                                verify_only=True
                            )
                hashed_layout = HashedLayout.from_layout(final_layout)
                if not valid:
                    continue
                if hashed_layout not in layout_table:
                    layout_table[hashed_layout] = (soln, score, size)
                else:
                    _, min_score, min_size = layout_table[hashed_layout]
                    if min_score < 0 or cmp(score, min_score, size, min_size):
                        layout_table[hashed_layout] = (soln, score, size)
                    else:
                        break  # This is a heap, so all remaining elements are worse.
            return [layout_table[hl][0] for hl in layout_table], None
        # Otherwise, try to merge from path collection.
        min_collection_list = []
        min_score = -1
    
        reduceable = [i for i in range(len(path_collection_list))]
        # Initialize priority queues for each operation in the front layer
        prio_queues = [PathPriorityQueue.buildheap(pc) for pc in path_collection_list]

        pjtree = PathJoinTree(
            [pq.peek()[0] for pq in prio_queues],
            self,
            kernel,
            future_gates
        )
        layout_table = {}
        for r in range(runs):
            # The root of the path join tree contains a potential solution
            # to the front layer.
            root = pjtree.root
            collection, is_valid, score, layout = root.data, root.valid, root.score, root.layout
            # Modify heaps and path join tree randomly at every step.
            # If our root is invalid, then find leaves contributing to the invalid nodes
            # and replace the corresponding folds with new ones.
            if not is_valid:
                self._dfs_find_invalid_and_update(prio_queues, pjtree)
                continue
            # Otherwise, randomly adjust the tree.
            # We perform a random adjustment using a fairness policy.
            # Specifically, we only choose an element from "reduceable".
            # After we choose this element, we remove it from reduceable.
            # When reduceable is empty, we repopulate it.
            x = np.random.randint(0, high=len(reduceable))
            ri = reduceable[x]
            pq = prio_queues[ri]
            path, score = pq.peek()
            pq.change_score(path, score+1)  # Push existing root of PQ down the heap
            new_path = pq.peek()[0]  # Replace that root with the new root in the path join tree
            pjtree.modify_leaves([ri], [new_path])
            del reduceable[x]  # Remove ri from reduceable as it has been sued
            if len(reduceable) == 0:
                reduceable = [i for i in range(len(path_collection_list))]
            # Get hash of solution and see if it is already used
            hashed_layout = HashedLayout.from_layout(layout)
            if hashed_layout not in layout_table:
                layout_table[hashed_layout] = (collection, score)
            else:
                min_collection, min_score = layout_table[hashed_layout]
                if score < min_score:
                    layout_table[hashed_layout] = (collection, score)
        for hl in layout_table:
            soln, score = layout_table[hl]
            min_collection_list.append(soln)
        # Now, we check if we have any solutions
        if len(min_collection_list) == 0:
            # If not, then reset all path priority queues.
            prio_queues = [PathPriorityQueue.buildheap(pc) for pc in path_collection_list]
            # Reset the path join tree
            modified_heaps = [i for i in range(len(path_collection_list))]
            new_paths = [pq.peek()[0] for pq in prio_queues]
            pjtree.modify_leaves(modified_heaps, new_paths)
            # Use DFS to find good splits in the front layer.
            dfs_stack = [pjtree.root]
            suggestions = []
            # Essentially, go down to the topmost valid node. Then, we keep all nodes
            # "answered" by that node in its own front layer. This is guaranteed to
            # have a solution as this node itself is a solution
            while dfs_stack:
                node = dfs_stack.pop()
                if node is None:
                    continue
                if not node.valid:
                    dfs_stack.append(node.left_child)
                    dfs_stack.append(node.right_child)
                else:
                    suggestions.append(node.target_index_list)
            return None, suggestions
        return min_collection_list, None

    def _remap_gate_for_layout(self, op, layout):
        new_op = copy(op)
        new_op.qargs = [self.canonical_register[layout[x]] for x in op.qargs]
        return new_op

    def _successors(self, node):
        for (_, s, edge_data) in self.input_dag.edges(node):
            if not isinstance(s, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield s

    def _path_minfold(self, kernel, path, future_gates):
        """
            Computes the minimum cost fold.

            path: the path to fold
            current_layout: the current running layout during routing
            post_primary_layer_view: a BFS-list of future operations that
                helps decide where to fold.

            Returns a list of equally optimal folds.
        """
        min_dist = -1
        min_folds = []
        latest_layout = kernel.layout
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
            else:  
                # if i != 0, then undo last backwards swap in last fold, 
                # and perform forwards swap prior to folded swap. 
                f0, f1 = path[i-1]
                b0, b1 = path[i]
                test_layout[b0], test_layout[b1] = test_layout[b1], test_layout[b0] # Flip back.
                test_layout[f0], test_layout[f1] = test_layout[f1], test_layout[f0] # Flip forward.
                fold.pop()  # Remove last swap.
                fold.insert(i-1, path[i-1])  # Insert new swap at beginning.
            # Update latest layout.
            latest_layout = test_layout
            latest_fold = fold
            # Compute distance to post primary layer.
            dist = self._distf(len(path)-1, future_gates, test_layout)
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

    def _verify_and_measure(
        self, 
        kernel,
        soln,
        future_gates,
        verify_only=False
    ):
        """
            During computation, due to the approximate nature of certain
            parts of ForeSight, we want to verify a given solution. We also
            do not want to waste precious cycles, so if we need to compute
            distance as well, we perform that in tandem.

            soln: a proposed solution
            target_list: the list of 2-qubit operations that must be satisfied
            current_layout: the current running layout during routing
            post_primary_layer_view: a BFS-list of future operations.

            Returns a tuple (IS_A_SOLN, cost, final_layout, size) where IS_A_SOLN is
                a bool indicating if the solution satisfies all operations in the
                front layer.
        """
        if len(kernel.front_layer) == 0:
            return True, 0
        test_layout = kernel.layout.copy()
        size = 0
        for layer in soln:
            for (p0,p1) in layer:
                test_layout.swap(p0, p1)
                size += 1
        max_allowed_size = 0
        for node in kernel.front_layer: 
            v0, v1 = node.qargs
            p0, p1 = test_layout[v0], test_layout[v1]
            if not self.coupling_map.graph.has_edge(p0, p1):
                return False, 0, test_layout, size
        # If we have gotten to this point, then our soln is good.
        if verify_only:
            dist = 0
        else:
            dist = self._distf(size, future_gates, test_layout)
        return True, dist, test_layout, size

    def _distf(self, soln_size, post_primary_layer_view, test_layout):
        """
            The distance heuristic function to determine the goodness of a
            solution.

            soln_size: total number of operations in the solution
            post_primary_layer_view: a BFS-list of future operations.
            test_layout: a trial layout used during routing that corresponds
                to the given solution

            Returns a float that is the distance value (lower is better)
        """
        dist = 0.0
        num_ops = 0
        for (node, depth) in post_primary_layer_view:
            depth_adjust = np.exp(-(depth/((self.mean_degree)**1.5))**2)
            if len(node.qargs) == 1 and self.noise_aware:
                q = node.qargs[0]
                p = test_layout[q]
                if node.name == 'measure':
                    w = self.ro_error_rates[p]
                else:
                    w = self.sq_error_rates[p]
                dist += w * depth_adjust
            else:
                q0, q1 = node.qargs
                p0, p1 = test_layout[q0], test_layout[q1]
                dist += self.distance_matrix[p0][p1] * depth_adjust
            num_ops += 1
        if num_ops == 0:
            return 0
        else:
            dist = dist/num_ops
            return dist+soln_size*np.exp(-(num_ops/(self.mean_degree**1.5))) 

    def _dfs_find_invalid_and_update(self, prio_queues, pjtree):
        dfs_stack = [pjtree.root]

        modified_heaps = []
        new_paths = []
        while dfs_stack:
            node = dfs_stack.pop()
            left, right = node.left_child, node.right_child
            if len(node.target_index_list) > 0:
                continue
            # Only go down invalid children
            # If neither are invalid, then update a random entry in the target_index_list.
            if (left is None or left.valid) and (right is None or right.valid):
                updated_index = int(np.random.choice(node.target_index_list, size=1))
                pq = prio_queues[updated_index]
                path, score = pq.peek()
                pq.change_score(path, score+1)  # Push this fold down the priority queue.
                # Now, replace this fold in the path join tree with the new
                # root of the heap.
                new_paths.append(pq.peek()[0])
                modified_heaps.append(updated_index)
            else:
                if left is not None and left.valid == 0:
                    dfs_stack.append(left)
                if right is not None and right.valid == 0:
                    dfs_stack.append(right)
        # Modify path join tree.
        if len(modified_heaps) > 0:
            pjtree.modify_leaves(modified_heaps, new_paths)
    
