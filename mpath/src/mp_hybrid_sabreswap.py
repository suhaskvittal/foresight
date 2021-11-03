# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# 
#
# MODIFICATION ALERT:
# This code has been modified by Suhas Vittal on date: 27 October 2021.
#
#

"""Routing via SWAP insertion using the SABRE method from Li et al."""

import logging
from collections import defaultdict
from copy import copy, deepcopy
import numpy as np

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode

from mp_layerview import LayerViewPass
import mp_hybrid_ips
from mp_util import G_MPATH_IPS_SOLN_CAP, G_MPATH_IPS_SLACK
from mp_stat import get_independent_variable

import warnings

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = 20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class MPATH_HYBRID_SabreSwap(TransformationPass):
    def __init__(self, coupling_map, regressor, heuristic="basic", seed=None, fake_run=False, metric_refresh_rate=10):
        """SabreSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
        """

        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.applied_predecessors = None
        self.qubits_decay = None
        self._bit_indices = None

        self.metric_refresh_rate = metric_refresh_rate
        self.regressor = regressor
        self.router_usage = defaultdict(int)

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        rng = np.random.default_rng(self.seed)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            mapped_dag = dag._copy_circuit_metadata()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = {qubit: 1 for qubit in dag.qubits}

        # Start algorithm from the front layer and iterate until all gates done.
        num_search_steps = 0
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)
        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1
        
        layer_view_pass = LayerViewPass()
        layer_view_pass.run(dag)
        primary_layer_view = layer_view_pass.property_set['primary_layer_view']
        secondary_layer_view = layer_view_pass.property_set['secondary_layer_view']

        original_layer_view_size = sum(len(layer) for layer in primary_layer_view)
        self.router_usage = defaultdict(int)

        number_of_swaps = 0
        exec_rest = False
        completed = set()
        while front_layer:
            if len(primary_layer_view) > 20 and (number_of_swaps + 1) % self.metric_refresh_rate == 0:
                # Update layer view.                 
                new_primary_layer_view = []
                new_secondary_layer_view = []
                for i in range(len(primary_layer_view)):
                    new_p_layer = []
                    new_s_layer = []
                    for node in primary_layer_view[i]:
                        if node not in completed:
                            new_p_layer.append(node)
                    for node in secondary_layer_view[i]:
                        if node not in completed:
                            new_s_layer.append(node)
                    if new_p_layer or new_s_layer:
                        new_primary_layer_view.append(new_p_layer)
                        new_secondary_layer_view.append(new_s_layer)
                primary_layer_view = new_primary_layer_view
                secondary_layer_view = new_secondary_layer_view
                # Now, determine if we stop using SABRE.
                X = get_independent_variable(primary_layer_view)
                y = self.regressor.predict(X)
                if y[0] < 0:  # Stop using SABRE.
                    exec_rest = True
                    break

            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    if self.coupling_map.graph.has_edge(current_layout[v0], current_layout[v1]):
                        execute_gate_list.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)

            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    completed.add(node)
                    front_layer.remove(node)
                    for successor in self._successors(node, dag):
                        self.applied_predecessors[successor] += 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                    if node.qargs:
                        self._reset_qubits_decay()

                # Diagnostics
                logger.debug("free! %s", [(n.name, n.qargs) for n in execute_gate_list])
                logger.debug("front_layer: %s", [(n.name, n.qargs) for n in front_layer])

                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            extended_set = self._obtain_extended_set(dag, front_layer)
            swap_candidates = self._obtain_swaps(front_layer, current_layout)
            swap_scores = dict.fromkeys(swap_candidates, 0)
            for swap_qubits in swap_scores:
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                score = self._score_heuristic(
                    self.heuristic, front_layer, extended_set, trial_layout, swap_qubits
                )
                swap_scores[swap_qubits] = score
            min_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
            best_swap = rng.choice(best_swaps)
            swap_node = DAGNode(op=SwapGate(), qargs=best_swap, type="op")
            self._apply_gate(mapped_dag, swap_node, current_layout, canonical_register)
            current_layout.swap(*best_swap)

            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE
            number_of_swaps += 1

            # Diagnostics
            logger.debug("SWAP Selection...")
            logger.debug("extended_set: %s", [(n.name, n.qargs) for n in extended_set])
            logger.debug("swap scores: %s", swap_scores)
            logger.debug("best swap: %s", best_swap)
            logger.debug("qubits decay: %s", self.qubits_decay)

        # Update router usage stats
        ips_layers_size = sum(len(layer) for layer in primary_layer_view)
        sabre_layers_size = original_layer_view_size - ips_layers_size
        self.router_usage['sabre'] = sabre_layers_size
        if exec_rest:
            # Build dag for the unfinished operations.
            post_dag = mapped_dag._copy_circuit_metadata()
            for i in range(len(primary_layer_view)):
                for node in primary_layer_view[i]:
                    self._apply_gate(post_dag, node, current_layout, canonical_register)
                for node in secondary_layer_view[i]:
                    self._apply_gate(post_dag, node, current_layout, canonical_register)
            # Run new routing algorithm on the post dag.
            ips = mp_hybrid_ips.MPATH_HYBRID_IPS(self.coupling_map, self.regressor, slack=G_MPATH_IPS_SLACK, solution_cap=G_MPATH_IPS_SOLN_CAP)
            post_dag = ips.run(post_dag)
            # Merge results.
            current_layout = ips.property_set['final_layout']
            warnings.filterwarnings("ignore")  # compose always gives many warnings.
            mapped_dag.compose(post_dag, inplace=True)
            # Update results
            self.router_usage['sabre'] += ips.router_usage['sabre']
            self.router_usage['ips'] += ips.router_usage['ips']

        self.property_set["final_layout"] = current_layout

        if not self.fake_run:
            return mapped_dag
        return dag


    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        if self.fake_run:
            return
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _reset_qubits_decay(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations).
        """
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _successors(self, node, dag):
        for _, successor, edge_data in dag.edges(node):
            if successor.type != "op":
                continue
            if isinstance(edge_data, Qubit):
                yield successor

    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.applied_predecessors[node] == len(node.qargs)

    def _was_resolved(self, node):
        return self.applied_predecessors[node] >= len(node.qargs)

    def _obtain_extended_set(self, dag, front_layer):
        """Populate extended_set by looking ahead a fixed number of gates.
        For each existing element add a successor until reaching limit.
        """
        extended_set = list()
        incremented = list()
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = list()
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    incremented.append(successor)
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in incremented:
            self.applied_predecessors[node] -= 1
        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swaps.add(tuple(swap))

        return candidate_swaps

    def _compute_cost(self, layer, layout):
        cost = 0
        for node in layer:
            cost += self.coupling_map.distance(layout[node.qargs[0]], layout[node.qargs[1]])
        return cost

    def _score_heuristic(self, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]])
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)



def _transform_gate_for_layout(op_node, layout, device_qreg):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)

    premap_qargs = op_node.qargs
    mapped_qargs = map(lambda x: device_qreg[layout[x]], premap_qargs)
    mapped_op_node.qargs = list(mapped_qargs)

    return mapped_op_node
