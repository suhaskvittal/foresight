"""
	author: Suhas Vittal
	date:	29 September 2021 @ 2:09 p.m. EST
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library import CXGate, SwapGate, Measure

from pulp import * 

import numpy as np

from copy import copy

from deepsolve import deep_solve

class MultipathSwap(TransformationPass):
	def __init__(self, coupling_map, seed=None, max_swaps=5, max_runs=1000, edge_weights=None):
		self.coupling_map = coupling_map		
		self.max_swaps = max_swaps
		self.max_runs = max_runs
		self.seed = seed

		self.requires = []
		self.preserves = []
			
	def run(self, dag):
		mapped_dag = dag._copy_circuit_metadata()
		canonical_register = dag.qregs["q"]
		current_layout = Layout.generate_trivial_layout(canonical_register)

		front_layer = dag.front_layer()
		finished = set()

		while len(front_layer) > 0:
			front_layer, current_layout, mapped_dag, finished =\
				deep_solve(
					self.coupling_map,
					dag,
					mapped_dag,
					current_layout,
					front_layer,
					finished,
					max_swaps=self.max_swaps
				)
		return mapped_dag
