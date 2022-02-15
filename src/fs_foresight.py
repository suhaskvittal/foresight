"""
    author: Suhas Vittal
    date:   29 September 2021
"""

from qiskit.circuit import Qubit

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.compiler import transpile
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import *
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from fs_layerview import LayerViewPass
from fs_selector import ForeSightSelector
from fs_sum_tree import SumTreeNode, MinLeafPackage
from fs_dist import process_coupling_map
from fs_multipath import ComputationKernel, DeepSolveSolution, ShallowSolveSolution
from fs_util import *

import numpy as np

from copy import copy, deepcopy
from collections import deque, defaultdict

class ForeSight(TransformationPass):
    def __init__(self,
        coupling_map, 
        slack=2,
        solution_cap=32,
        asap_boost=False,
        asap_only=False,
        approx_asap=False,
        edge_weights=None,
        vertex_weights=None,
        readout_weights=None,
        debug=False
    ):
        """
        Initializes ForeSight pass.

        coupling_map: backend coupling graph
        slack: expands path consideration. If slack=0, then shortest paths are
            considered. If slack=1, then shortest and second shortest paths are 
            considered. (etc.)
        solution_cap: caps size of the computation tree. Tree is contracted
            if treewidth exceeds 2*solution_cap.
        edge_weights: weights for noisy computation.
            vertex_weights: weights for noisy computation. (UNUSED)
            readout_weights: weights for noisy computation. (UNUSED)
            debug: output debug messages 
        """
        super().__init__()

        self.slack = slack
        self.coupling_map = coupling_map        
        self.solution_cap = solution_cap

        self.distance_matrix, self.paths_on_arch = process_coupling_map(
            coupling_map,
            slack,
            edge_weights=edge_weights,
        ) 
        self.mean_degree = len(self.coupling_map.get_edges()) / self.coupling_map.size()
        self.use_asap_boost = asap_boost
        self.approx_asap = approx_asap
        self.use_asap_only = asap_only

        # Other
        self.fake_run = False
        self.debug = debug

        self.base_pred = defaultdict(int)

        # initialize weights
        self.edge_weights = edge_weights
        self.vertex_weights = vertex_weights
        self.readout_weights = readout_weights

        # usage statistics
        self.suggestions = 0
        self.swap_segments = []
        self.alap_used = 0
        self.asap_used = 0
            
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
                (1) If this is normal ForeSight (no ASAP boost),
                    then, we verify in two places.
                        (a) Shallow Solve -- we verify that all
                            2-qubit operations in the layer have
                            adjacent operands.
                        (b) In this method -- we verify that all
                            operations in the original dag have
                            been placed.
                (2) If this is ASAP boosted ForeSight, then we
                    verify that all operations in the original
                    dag have been placed and verify that all
                    2-qubit operations are valid in this method.
                    For checking that all 2-qubit operations are
                    valid, we run SABRE to check if any new SWAPs
                    are added. If no such SWAPs are found, then 
                    all operands of 2-qubit operations are adjacent.
        """
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

        self.base_pred = defaultdict(int)
        for (_,input_node) in dag.input_map.items():
            for s in self._successors(input_node,dag):
                self.base_pred[s] += 1
        # Run deep solve
        front_layer = []
        front_layer.extend(primary_layer_view[0])
        front_layer.extend(secondary_layer_view[0])
        solutions = []
        if not self.use_asap_only:
            solutions.append(DeepSolveSolution([], current_layout, 0, set(), front_layer, False, kernel_type='alap')) 
        if self.use_asap_boost:
            solutions.append(DeepSolveSolution([],current_layout,0,set(),front_layer,False,kernel_type='asap'))
        while primary_layer_view:
            solutions = self.deep_solve(
                dag,
                primary_layer_view, 
                secondary_layer_view, 
                solutions,
                canonical_register 
            )

        # Choose solution with minimum depth.
        min_dag = None
        min_layout = None
        min_size = -1
        min_depth = -1
        min_segments = None
        min_alap_used = 0
        min_asap_used = 0
        min_type = ''
        for deep_solve_soln in solutions:
            layout, soln, _ =\
                deep_solve_soln.layout, deep_solve_soln.output_layers, deep_solve_soln.layer_sum
            test_dag = dag._copy_circuit_metadata()
            for layer in soln:
                for node in layer:
                    test_dag.apply_operation_back(op=node.op, qargs=node.qargs, cargs=node.cargs)
            test_circ = dag_to_circuit(test_dag)
            # Run qiskit opt 3
            test_circ = transpile(
                test_circ,
                basis_gates=G_QISKIT_GATE_SET,
                coupling_map=self.coupling_map,
                layout_method='trivial',
                routing_method='none',
                optimization_level=3,
                approximation_degree=1
            )
            size = test_circ.size()
            depth = test_circ.depth()
            if min_dag is None or (size < min_size) or (size == min_size and depth < min_depth):
                min_layout = layout
                min_dag = test_dag
                mapped_dag = circuit_to_dag(test_circ)
                min_size = size
                min_depth = depth
                min_segments = deep_solve_soln.swap_segments
                min_alap_used = deep_solve_soln.alap_used
                min_asap_used = deep_solve_soln.asap_used
                min_type = deep_solve_soln.type
        self.swap_segments = min_segments
        self.alap_used = min_alap_used
        self.asap_used = min_asap_used
        self.property_set['final_layout'] = min_layout
        print('solution is ', min_type)

        if self.fake_run:
            return mapped_dag   
        # Validate mapped dag (SWAPs already validated -- ops are valid, just check if all ops are there)
        orig_ops = dag.count_ops()
        mapped_ops = min_dag.count_ops()
        # Self-verify that no operations have been skipped.
        for g in orig_ops:
            if g not in mapped_ops or orig_ops[g] != mapped_ops[g]:
                print('Error! Unequal non-SWAP gates after routing.')
                print('Original circuit: ', orig_ops)
                print('Mapped circuit: ', mapped_ops)
                exit()
        if self.debug:
            print('Statistics')
            print('\tSuggestion usage:', self.suggestions)
        if self.use_asap_boost:  # ASAP boosted routing is hard to verify, so do so here.
            # If SABRE adds swaps to our output dag, then we know we have mis-routed.
            remapped_dag = SabreSwap(self.coupling_map).run(min_dag)
            remapped_ops = remapped_dag.count_ops()
            if 'swap' in remapped_ops and 'swap' not in mapped_ops:
                print('Error! Remapped dag contains extra SWAPs.')
                print('Mapped circuit: ', mapped_dag['swap'])
                print('Remapped circuit: ', remapped_dag['swap'])
                exit()
            if 'swap' in remapped_ops and 'swap' in mapped_ops and remapped_ops['swap'] != mapped_ops['swap']:
                print('Error! Remapped dag contains extra SWAPs.')
                print('Mapped circuit: ', mapped_dag['swap'])
                print('Remapped circuit: ', remapped_dag['swap'])
                exit()
        return mapped_dag
    
    def deep_solve(
        self, 
        dag,
        primary_layer_view, 
        secondary_layer_view, 
        base_solver_queue, 
        canonical_register 
    ):
        """
            Deep solve portion. Maintains the computation tree. Deep
            solve maintains a list of ComputationKernels in its
            solver queue (each represent a potential solution). Once
            the queue of computation kernels grows to large, Deep Solve
            traverse upwards to the original computation kernel and 
            collapse all the data down to the current kernels (contracting
            the computation tree). Once this finishes, deep solve will
            call itself to restart the process with the remaining
            computation kernels as the new original kernels.

            This process can be represented as a tree, where each
            computation kernel is a node in the tree. Then, the entire
            process of contracting the tree traverses up to the root and
            "flattens" the tree to the kernels at the leaves.

            Each solution from deep solve contains
                (1) A list of operations in a BFS-like list.
                (2) The number of SWAPs in the list.
                (3) A current layout.
                (4) The completed nodes set.
            This is identical to a ShallowSolveSolution, but we use a 
            different name to differentiate the usage of both solutions.

            dag: the base circuit's dag representation.
            primary_layer_view: a BFS view of all 2-qubit operations
            secondary_layer_view: a BFS view of all other operations
            base_solver_queue: holds current solutions for routing
            canonical_register: necessary for routing correctly

            Returns: a list of DeepSolveSolutions.
        """
        if len(primary_layer_view) == 0:
            return base_solver_queue
        # Build tree of output layers.
        solver_queue = []
        leaves = []
        for (i, deep_solve_soln) in enumerate(base_solver_queue):
            root_node = SumTreeNode(
                deep_solve_soln.output_layers,
                deep_solve_soln.layer_sum,
                None,
                [],
                swap_segments=deep_solve_soln.swap_segments,
                alap_used=deep_solve_soln.alap_used,
                asap_used=deep_solve_soln.asap_used
            )  
            leaves.append(root_node)
            # Each solution instance is a "computation kernel".
            # We perform deep solve on a tree of kernels by performing
            # shallow solve until some critical point.
            solver_queue.append(ComputationKernel(
                deep_solve_soln.layout,
                i,
                deep_solve_soln.completed_nodes,
                deep_solve_soln.last_front_layer,
                deep_solve_soln.is_dirty,
                kernel_type=deep_solve_soln.type,
            ))
        while len(solver_queue) <= 2*self.solution_cap:
            if len(primary_layer_view) == 0:
                break
            next_solver_queue = []
            next_leaves = []
            for compkern in solver_queue:  # Empty out queue into next_solver_queue.
                parent = leaves[compkern.parent_id]
                if compkern.type == 'asap':
                    solutions = self.asap_burst_solve(
                        dag,
                        primary_layer_view,
                        secondary_layer_view,
                        compkern.last_front_layer,
                        compkern.layout,
                        canonical_register,
                        compkern.completed_nodes
                    )
                else:
                    solutions = self.shallow_solve(
                        dag,
                        primary_layer_view,
                        secondary_layer_view,
                        compkern.last_front_layer,
                        compkern.layout,
                        canonical_register,
                        compkern.completed_nodes
                    )
                # Apply solutions non-deterministically to current_dag.
                for (i, shallow_solve_soln) in enumerate(solutions):
                    # Create node for each candidate solution.
                    num_children = 0
                    for kernel_type in ['asap','alap']:
                        if shallow_solve_soln.num_swaps == 0 and kernel_type != compkern.type:
                            continue
                        if kernel_type == 'asap' and not self.use_asap_boost:
                            continue
                        if kernel_type == 'alap' and self.use_asap_only:
                            continue
#                        if self.approx_asap and compkern.type != kernel_type and np.random.random() > 0.25:
#                            continue
#                        if compkern.type == 'alap' and kernel_type == 'asap' and compkern.is_dirty:
#                            continue
                        is_dirty = (kernel_type == 'asap' and compkern.type == 'alap') or compkern.is_dirty
                        node = SumTreeNode(
                            shallow_solve_soln.output_layers,
                            parent.sum_data + shallow_solve_soln.num_swaps,
                            parent,
                            [],
                            swap_segments=[shallow_solve_soln.num_swaps],
                            alap_used=(shallow_solve_soln.num_swaps\
                                if compkern.type == 'alap' and shallow_solve_soln.num_swaps > 0 else 0)\
                                    +parent.alap_used,
                            asap_used=(shallow_solve_soln.num_swaps\
                                if compkern.type == 'asap' and shallow_solve_soln.num_swaps > 0 else 0)\
                                    +parent.asap_used,
                        )
                        parent.children.append(node)
                        next_leaves.append(node)
                        # Create a child kernel
                        next_solver_queue.append(ComputationKernel(
                            shallow_solve_soln.layout,
                            len(next_leaves) - 1,
                            shallow_solve_soln.completed_nodes,
                            shallow_solve_soln.last_front_layer,
                            is_dirty,
                            kernel_type=kernel_type,
                        ))
                        num_children += 1
                    if num_children == 0:
                        print('ERROR: ZERO CHILDREN')
                        exit()
            solver_queue = next_solver_queue
            leaves = next_leaves
            # Remove top layer
            for node in primary_layer_view[0]:
                for s in self._successors(node,dag):
                    self.base_pred[s] += 1
            for node in secondary_layer_view[0]:
                for s in self._successors(node,dag):
                    self.base_pred[s] += 1
            primary_layer_view.popleft()
            secondary_layer_view.popleft()
        # The critical point has been reached -- we contract the computation tree.
        # Now, we simply check the leaves of the output layer tree. We select the leaf with the minimum sum.
        min_leaves = defaultdict(list)
        min_cnots = {}
        for (i, leaf) in enumerate(leaves):
            compkern = solver_queue[i]
            kernel_type = 'asap' if leaf.asap_used >= leaf.alap_used else 'alap'
            if kernel_type not in min_cnots or leaf.sum_data < min_cnots[kernel_type]:
                min_leaves[kernel_type] = [MinLeafPackage(
                    leaf,
                    leaf.sum_data,
                    compkern.layout,
                    compkern.last_front_layer,
                    compkern.is_dirty,
                    compkern.completed_nodes,
                    leaf.alap_used,
                    leaf.asap_used,
                    compkern.type
                )] 
                min_cnots[kernel_type] = leaf.sum_data
            elif leaf.sum_data == min_cnots[kernel_type]:
                min_leaves[kernel_type].append(MinLeafPackage(
                    leaf,
                    leaf.sum_data,
                    compkern.layout,
                    compkern.last_front_layer,
                    compkern.is_dirty,
                    compkern.completed_nodes,
                    leaf.alap_used,
                    leaf.asap_used,
                    compkern.type
                )) 
        # If we have too many minleaves, then randomly choose a subset of them.
        adjusted_cap = self.solution_cap // len(min_leaves.keys())
        min_solutions = []
        for policy in min_leaves:
            if len(min_leaves[policy]) > adjusted_cap:
                min_leaf_indices = np.random.choice(np.arange(len(min_leaves[policy])), size=adjusted_cap)
                min_leaves[policy] = [min_leaves[policy][i] for i in min_leaf_indices]
            # Now that we know the best leaves, we simply just need to build the corresponding dags.
            # Traverse up the leaves -- there is only one path to a root.
            for min_leaf in min_leaves[policy]:
                output_layer_deque = deque([])
                curr = min_leaf.leaf_node
                swap_segments = []  # build swap segments from the tree
                while curr is not None:
                    output_layer_deque.extendleft(list(curr.obj_data)[::-1])
                    if len(curr.swap_segments) > 0 and curr.swap_segments[0] != 0:
                        swap_segments.extend(curr.swap_segments)
                    curr = curr.parent
                # Package the solution as a DeepSolveSolution
                min_solutions.append(DeepSolveSolution(
                    output_layer_deque,
                    min_leaf.layout,
                    min_leaf.leaf_sum,
                    min_leaf.completed_nodes,
                    min_leaf.last_front_layer,
                    min_leaf.is_dirty,
                    swap_segments=swap_segments[::-1],  # it is backwards because we went from leaf to root
                    alap_used=min_leaf.alap_used,
                    asap_used=min_leaf.asap_used,
                    kernel_type=min_leaf.type
                ))
        return min_solutions
                
    def shallow_solve(
        self, 
        dag,
        primary_layer_view,
        secondary_layer_view,
        prev_front_layer,
        current_layout,
        canonical_register,
        completed_nodes
    ):
        """
            Performs actual routing and node marking.

            Shallow solve first identifies unsatisfied 2-qubit
            operations in the layer and then tries to find SWAPs
            that minimize depth and size. If there isn't a solution
            that satisfies all unsatisfied 2-qubit operations, then
            shallow solve splits up the layer into multiple layers
            depending on analysis of the concurrent satisfiability 
            (this is called "suggestion"). Then, it will run greedily
            on these layers and stitch their results together. 

            Regardless of whether the suggestion phase occurs or not,
            shallow solve will return a list of ShallowSolveSolutions
            that represent routing solutions to the topmost layer. Each
            solution contains
                (1) A list of operations in a BFS-like list.
                (2) The number of SWAPs in the list.
                (3) A new layout.
                (4) A completed set.

            primary_layer_view: BFS view of 2-qubit operations.
            secondary_layer_view: BFS view of other operations.
            current_layout: current running layout
            canonical_register: necessary for routing correctly
            completed_nodes: a set of nodes that have already been routed

            Returns: a list of ShallowSolveSolutions that route the current
                layer given the completed_nodes set.
        """
        # Initialize all structures
        output_layers = [[]]
        
        target_list = []
        path_collection_list = []
        post_nodes = []
        target_to_node = {}

        # Copy the set of completed nodes
        # as the set will be used across multiple
        # computation kernels.
        completed_nodes = copy(completed_nodes)

        # Only execute operations in current layer.
        # Define post primary layer view
        post_primary_layer_view = self.compute_post_primary(
            primary_layer_view, 
            completed_nodes
        )
    
        # Process operations in layer
        for node in secondary_layer_view[0]:
            if node in completed_nodes:
                continue
            output_layers[0].append(self._remap_gate_for_layout(node, current_layout, canonical_register))
            completed_nodes.add(node)  # preemptively add the node -- we will place it anyways.
        front_layer = primary_layer_view[0]
        for node in front_layer:  
            if node in completed_nodes:
                continue
            q0, q1 = node.qargs
            # Filter out all operations that are currently adjacent.
            if self.coupling_map.graph.has_edge(current_layout[q0], current_layout[q1]):
                output_layers[0].append(self._remap_gate_for_layout(node, current_layout, canonical_register))
                completed_nodes.add(node)
            else:  # Otherwise, get path candidates and place it in path_collection_list.
                path_collection_list.append(self.path_find_and_fold(q0, q1, post_primary_layer_view, current_layout))
                post_nodes.append(node)
                target_list.append((q0, q1))
                target_to_node[(q0, q1)] = node
        # Build PPC and get candidate list.
        if len(path_collection_list) == 0:
            return [ShallowSolveSolution(output_layers, current_layout, 0, completed_nodes, prev_front_layer)]
        # This part is pretty much described as "magic" in the paper.
        # This step is heavily rooted in theory for its correctness, so we excluded the
        # explanation.
        path_selector = ForeSightSelector(path_collection_list, len(path_collection_list))
        candidate_list, suggestions = path_selector.find_and_join(self, target_list, current_layout, post_primary_layer_view)
        if candidate_list is None:  # We failed, take the suggestions.
            self.suggestions += 1
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
                tmp_pl_view.appendleft([target_to_node[target] for target in target_sub_list])
                tmp_sl_view.appendleft([])
            solutions = [ShallowSolveSolution(output_layers, current_layout, 0, completed_nodes, prev_front_layer)]
            while target_lists:
                target_sub_list = target_lists.pop()
                next_solutions = []
                for prev_soln in solutions:
                    solution_list = self.shallow_solve(
                        dag,
                        tmp_pl_view,
                        tmp_sl_view,
                        prev_front_layer,
                        prev_soln.layout,
                        canonical_register,
                        prev_soln.completed_nodes
                    )
                    for new_soln in solution_list:
                        prev_layers_cpy = copy(prev_soln.output_layers)
                        prev_layers_cpy.extend(new_soln.output_layers)
                        next_solutions.append(ShallowSolveSolution(
                            prev_layers_cpy,
                            new_soln.layout,
                            prev_soln.num_swaps+new_soln.num_swaps,
                            new_soln.completed_nodes,
                            prev_front_layer,
                        ))
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
                    swp_gate = DAGOpNode(
                        op=SwapGate(),
                        qargs=[v0, v1]  
                    )
                    output_layers_cpy[-1].append(self._remap_gate_for_layout(swp_gate, new_layout, canonical_register))
                    # Apply swap to modify running layout.
                    new_layout.swap(p0, p1)
                    num_swaps += 1
            # Apply the operations after completing all the swaps.
            output_layers_cpy.append([])
            for node in post_nodes:  
                q0, q1 = node.qargs
                # Self-verify that proper routing is occurring.
                if not self.coupling_map.graph.has_edge(new_layout[q0], new_layout[q1]):
                    print('ERROR: not satisified %d(%d), %d(%d)'\
                        % (q0.index, current_layout[q0], q1.index, current_layout[q1]))
                    print('%d edges:' % current_layout[q0], self.coupling_map.neighbors(current_layout[q0]))
                    print('%d edges:' % current_layout[q1], self.coupling_map.neighbors(current_layout[q1]))
                    print('|TargetSet| = %d' % len(target_list))
                    print('Solution: ', soln)
                    exit()
                output_layers_cpy[-1].append(self._remap_gate_for_layout(node, new_layout, canonical_register))
            solutions.append(ShallowSolveSolution(
                output_layers_cpy,
                new_layout,
                num_swaps,
                completed_nodes,
                prev_front_layer
            ))
        for node in front_layer:
            completed_nodes.add(node)
        return solutions

    def asap_burst_solve(
        self, 
        dag,
        primary_layer_view,
        secondary_layer_view,
        prev_front_layer,
        current_layout,
        canonical_register,
        completed_nodes
    ):
        """
            Extension for ForeSight where we use ASAP routing to
            map gates onto a hardware backend. The ASAP routing lasts
            for up to "burst_size" 2-qubit operations and then
            returns a list of SWAPs.

            The ASAP policy here is just a modified version of SABRE
            that works with ForeSight. Future work could on designing
            new ASAP policies.

            In theory, any policy can be integrated with ForeSight, as we
            have done here.

            dag: the input dag
            primary_layer_view: BFS list of 2-qubit operations
            secondary_layer_view: BFS list of 1-qubit and barrier operations
            current_layout: running layout before calling ASAP
            canonical_register: required for gate mapping
            completed_nodes: a list of dag nodes that have been routed
            burst_size: the number of 2-qubit operations to route before returning

            Returns an array containing a single ShallowSolveSolution object.
        """
        output_layers = [] 
        completed_nodes = copy(completed_nodes)

        front_layer = []
        front_layer.extend(primary_layer_view[0])
        front_layer.extend(secondary_layer_view[0])
        # setup predecessor table
        pred = defaultdict(int)
        #for x in self.base_pred:
        #    pred[x] = self.base_pred[x]
        # copy layout
        new_layout = current_layout.copy()
        completion_count = 0
        num_swaps = 0
        while front_layer:
            output_layers.append([])
            exec_list = []
            next_front_layer = []
            for node in front_layer:
                if len(node.qargs) != 2 or node in completed_nodes: 
                    exec_list.append(node)
                else:
                    q0,q1 = node.qargs
                    if self.coupling_map.graph.has_edge(new_layout[q0],new_layout[q1]):
                        exec_list.append(node)
                    else:
                        next_front_layer.append(node)
            if exec_list:
                for node in exec_list:
                    for s in self._successors(node,dag):
                        if s not in pred:
                            pred[s] = self.base_pred[s]
                        pred[s] += 1
                        if pred[s] == len(s.qargs):
                            next_front_layer.append(s)
                    if node not in completed_nodes:
                        output_layers[-1].append(
                            self._remap_gate_for_layout(node, new_layout, canonical_register)
                        )
                        if len(node.qargs) == 2:
                            completion_count += 1
                        completed_nodes.add(node)
                front_layer = next_front_layer
                continue
            if all(x in completed_nodes for x in front_layer):
                front_layer = next_front_layer
                break
            # Nothing is executable, try to swap some qubits.
            # Setup root set
            root_set = set()
            for node in next_front_layer:   
                q0,q1 = node.qargs
                root_set.add(new_layout[q0])
                root_set.add(new_layout[q1])
            # Create post primary layer view
            min_swaps = []
            min_score = -1
            for s1 in root_set:
                for s2 in self.coupling_map.neighbors(s1):
                    test_layout = new_layout.copy() 
                    test_layout.swap(s1,s2)
                    score = self._asap_distf(dag, next_front_layer, pred, test_layout,\
                                completed_nodes)
                    if score < min_score or min_score == -1:
                        min_swaps = [(s1,s2)]
                        min_score = score
                    elif score == min_score:
                        min_swaps.append((s1,s2))
            # Randomly choose one swap.
            (s1,s2) = min_swaps[np.random.randint(0,high=len(min_swaps))]
            swp_gate = DAGOpNode(
                op=SwapGate(),
                qargs=[new_layout[s1],new_layout[s2]]
            ) 
            num_swaps += 1
            output_layers[-1].append(
                self._remap_gate_for_layout(swp_gate, new_layout, canonical_register)
            )
            new_layout.swap(s1,s2)
            front_layer = next_front_layer
        # Make sure all operations in front layer are completed (primary_layer_view[0])
        return [ShallowSolveSolution(
            output_layers,
            new_layout,
            num_swaps,
            completed_nodes,
            front_layer
        )]

    def compute_post_primary(self, primary_layer_view, completed_nodes, depth_unaware=False):
        """
            Computes the post primary layer view. A post primary layer
            view (PPLV) is a subset of the primary layer view that contains
            2-qubit operations such that for both qubits used in the 
            operation, this is their first usage in the PPLV.
            
            primary_layer_view: the BFS-list of 2-qubit operations in the DAG
            completed_nodes: a set of already-routed operations.

            Returns a BFS-list of future operations.
        """
        if len(primary_layer_view) == 1:
            post_primary_layer_view = []
        else:
            post_primary_layer_view = []
            visited = set()
            # The post primary layer is composed the first 2-qubit operation for each qubit
            # in the circuit. The maximum size of the post primary layer is NQUBITS/2.
            for i in range(1, min(len(primary_layer_view), int(np.ceil(10*self.mean_degree)))):
                curr_layer = []
                for node in primary_layer_view[i]:
                    # If the node is already completed, do not consider it for the
                    # heuristic
                    if node in completed_nodes:
                        continue
                    # Otherwise, add it
                    q0, q1 = node.qargs
                    if q0 in visited or q1 in visited:
                        visited.add(q0)
                        visited.add(q1)
                        continue
                    visited.add(q0)
                    visited.add(q1)
                    curr_layer.append(node)
                if primary_layer_view[i]:
                    if (depth_unaware and curr_layer) or not depth_unaware:
                        post_primary_layer_view.append(curr_layer)
        return post_primary_layer_view

    def path_find_and_fold(self, v0, v1, post_primary_layer_view, current_layout):
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
            dist = self._alap_distf(len(path)-1, post_primary_layer_view, test_layout)
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

    def _successors(self, node, dag):
        for (_, successor, edge_data) in dag.edges(node):
            if not isinstance(successor, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield successor

    def _verify_and_measure(
        self, 
        soln,
        target_list,
        current_layout,
        post_primary_layer_view,
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

            Returns a tuple (IS_A_SOLN, dist) where IS_A_SOLN is a boolean and
                dist is a float.
        """
        if len(target_list) == 0:
            return True, 0
        test_layout = current_layout.copy()
        size = 0
        flattened_soln = []
        for layer in soln:
            for (p0, p1) in layer:
                test_layout.swap(p0, p1)
                flattened_soln.append((p0,p1))
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
            dist = self._alap_distf(size, post_primary_layer_view, test_layout)
        return True, dist

    def _alap_distf(self, soln_size, post_primary_layer_view, test_layout):
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
        for r in range(0, len(post_primary_layer_view)):
            post_layer = post_primary_layer_view[r]
            sub_sum = 0.0
            for node in post_layer:
                q0, q1 = node.qargs
                p0, p1 = test_layout[q0], test_layout[q1]
                num_ops += 1
                s = self.distance_matrix[p0][p1]
                sub_sum += s
            dist += sub_sum * np.exp(-(r/((self.mean_degree)**1.5))**2)
        if num_ops == 0:
            return 0
        else:
            dist = dist/num_ops
            return dist+soln_size*np.exp(-(num_ops/(self.mean_degree**1.5))) 
    
    def _asap_distf(self, dag, front_layer, pred, test_layout, completed_nodes, explore_size=20): 
        """
            The lookahead distance heuristic used in SABRE.

            dag: the input dag
            front_layer: the current top layer that is being routed
            pred: the predecessor table for each dag node
            test_layout: layout to evaluate
            completed_nodes: a set of dag nodes that have been routed
            explore_size: number of dag nodes to consider for layout evaluation

            Returns a float that is the distance value (lower is better).
        """
        inc = []
        tmp_front = front_layer
        
        dist1,dist2 = 0,0
        explored = 0
        done = False
        for node in front_layer:
            v0,v1 = node.qargs
            p0,p1 = test_layout[v0],test_layout[v1]
            dist1 += self.distance_matrix[p0][p1]
        dist1 /= len(front_layer)
        while tmp_front and not done:
            next_front = []
            for node in tmp_front:
                for s in self._successors(node,dag):
                    inc.append(s)
                    if node not in completed_nodes:
                        pred[s] += 1
                    if pred[s] == len(s.qargs):
                        next_front.append(s)
                        if len(s.qargs) == 2 and s not in completed_nodes:
                            v0,v1 = s.qargs
                            p0,p1 = test_layout[v0],test_layout[v1]
                            dist2 += 0.5*self.distance_matrix[p0][p1]
                            explored += 1 
                if explored >= explore_size:
                    done = True
                    break
            tmp_front = next_front
        for node in inc:
            pred[node] -= 1
        if explored > 0:
            dist2 /= explored
        return dist1+dist2
    
