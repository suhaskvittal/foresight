"""
    author: Suhas Vittal
    date:   7 January 2022
"""

from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGOpNode

from collections import defaultdict
from copy import copy

class Excavator:
    def __init__(self, front_layer, dag):
        self._pred = defaultdict(int)
        self.front_layer = copy(front_layer)
        self.dag = dag

        for x in front_layer:
            self._pred[x] = len(x.qargs)

    def excavate(self, current_layout, paths_on_arch, df=2):
        usage_map = {x: self._get_usage_space(x, current_layout, paths_on_arch, df)\
                        for x in front_layer}
        intersection_score = {x: 0 for x in front_layer}
        # Compute the total number of intersections with other usage spaces
        for i in range(len(front_layer)):
            x = front_layer[i]
            for j in range(i+1,len(front_layer)): 
                y = front_layer[j]
                number_of_intersects = len(usage_map[x].intersection(usage_map[y]))
                intersection_score[x] += number_of_intersects
                intersection_score[y] += number_of_intersects
        # Return the node that has the most intersections.
        return max(front_layer, key=lambda x: usage_map[x])

    def _get_oplist(self, node):
        # oplist is a list of all immediately satisfiable ops
        # conditioned on node being satisfied.
        oplist = []
        pred_copy = copy(self._pred)
        
        exec_list = [node]
        while exec_list:
            parent = exec_list.pop()
            oplist.append(parent)

            for (_,x,edge_data) in self.dag.edges(x):
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
            for i in range(df):
                path = paths_p0p1[i]
                for (r0,r1) in path:
                    usage_space.add(r0)
                    usage_space.add(r1)
        return usage_space

