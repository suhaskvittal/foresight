"""
    author: Suhas Vittal
    date:   7 January 2022
"""

from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGOpNode

from collections import defaultdict
from copy import copy

class ForeSightExcavator:
    def __init__(self, front_layer, dag, prev_completed_nodes, critical_point=2):
        self._pred = defaultdict(int)
        self.front_layer = copy(front_layer)
        self.dag = dag
        self.critical_point = critical_point

        for x in prev_completed_nodes:
            for (_,y,edge_data) in self.dag.edges(x):
                if not (isinstance(y, DAGOpNode) and isinstance(edge_data,Qubit)):
                    continue
                self._pred[y] += 1

    def excavate(self, current_layout, paths_on_arch, df=3):
        if len(self.front_layer) < self.critical_point:
            return None

        usage_map = {x: self._get_usage_space(x, current_layout, paths_on_arch, df)\
                        for x in self.front_layer}
        intersection_score = {x: 0 for x in self.front_layer}
        # Compute the total number of intersections with other usage spaces
        for i in range(len(self.front_layer)):
            x = self.front_layer[i]
            for j in range(i+1,len(self.front_layer)): 
                y = self.front_layer[j]
                number_of_intersects = len(usage_map[x].intersection(usage_map[y]))
                intersection_score[x] += number_of_intersects
                intersection_score[y] += number_of_intersects
        # Get the node that has the most intersections.
        max_node_index = 0
        max_node = self.front_layer[0]
        for i in range(1, len(self.front_layer)):
            node = self.front_layer[i]
            if intersection_score[max_node] < intersection_score[node]:
                max_node_index = i
                max_node = node
        if intersection_score[max_node] == 0:
            return None
        del self.front_layer[i]
        oplist = self._get_oplist(max_node)
        for node in oplist:
            for (_,x,edge_data) in self.dag.edges(node):
                if not (isinstance(x, DAGOpNode) and isinstance(edge_data,Qubit)):
                    continue
                self._pred[x] += 1
                if len(x.qargs) == 2\
                and self._pred[x] == len(x.qargs)\
                and x not in oplist:
                    self.front_layer.append(x)
        return oplist

    def _get_oplist(self, node):
        # oplist is a list of all immediately satisfiable ops
        # conditioned on node being satisfied.
        oplist = []
        pred_copy = copy(self._pred)
        
        exec_list = [node]
        while exec_list:
            parent = exec_list.pop()
            oplist.append(parent)

            for (_,x,edge_data) in self.dag.edges(parent):
                if not (isinstance(x, DAGOpNode) and isinstance(edge_data,Qubit)):
                    continue
                pred_copy[x] += 1
                if pred_copy[x] == len(x.qargs):
                    exec_list.append(x)
        return oplist

    def _get_usage_space(self, node, current_layout, paths_on_arch, df):
        oplist = self._get_oplist(node) 

        usage_space = set()
        # Essentially, we compute the usage space as all the
        # physical qubits visited by satisfying 2-qubit ops
        # in the oplist. We assume we use the shortest path
        # for satisfying the op -- the degree of freedom
        # parameter also considers qubits visited by using
        # longer paths (1 = only shortest path, 2 = shortest
        # path and next shortest path, etc.)
        for x in oplist:
            if len(x.qargs) == 1:
                continue
            q0,q1 = x.qargs
            p0,p1 = current_layout[q0], current_layout[q1]
            paths_p0p1 = paths_on_arch[(p0,p1)]
            for i in range(min(df, len(paths_p0p1))):
                path = paths_p0p1[i]
                for (r0,r1) in path:
                    usage_space.add(r0)
                    usage_space.add(r1)
        return usage_space

