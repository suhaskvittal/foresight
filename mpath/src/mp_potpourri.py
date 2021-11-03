"""
    author: Suhas Vittal
    date:   2 November 2021
"""

from qiskit.transpiler.passes import *
from qiskit.transpiler import CouplingMap, PassManager

from mp_exec import _pad_circuit_to_fit
from mp_util import G_QAOA,\
                    G_IBM_TORONTO,\
                    G_RIGETTI_ASPEN9,\
                    G_GOOGLE_WEBER,\
                    G_QISKIT_GATE_SET

import pandas as pd

from collections import defaultdict

def get_sk_model_trend(): 
    suite = G_QAOA

    toronto_sabre = PassManager([
        SabreLayout(G_IBM_TORONTO, routing_pass=SabreSwap(G_IBM_TORONTO, heuristic='decay')),
        ApplyLayout(),
        SabreSwap(G_IBM_TORONTO, heuristic='decay'),
        Unroller(G_QISKIT_GATE_SET)
    ])
    weber_sabre = PassManager([
        SabreLayout(G_GOOGLE_WEBER, routing_pass=SabreSwap(G_GOOGLE_WEBER, heuristic='decay')),
        ApplyLayout(),
        SabreSwap(G_GOOGLE_WEBER, heuristic='decay'),
        Unroller(G_QISKIT_GATE_SET)
    ])
    aspen9_sabre = PassManager([
        SabreLayout(G_RIGETTI_ASPEN9, routing_pass=SabreSwap(G_RIGETTI_ASPEN9, heuristic='decay')),
        ApplyLayout(),
        SabreSwap(G_RIGETTI_ASPEN9, heuristic='decay'),
        Unroller(G_QISKIT_GATE_SET)
    ])

    data = defaultdict(list)
    for (_, _, circ) in suite:
        n_qubits = circ.num_qubits 
        og_cx = circ.count_ops()['cx']
        # Toronto
        toronto_circ = circ.copy()
        _pad_circuit_to_fit(toronto_circ, G_IBM_TORONTO)
        toronto_circ = toronto_sabre.run(toronto_circ)
        toronto_cx = toronto_circ.count_ops()['cx']
        # Weber
        weber_circ = circ.copy()
        _pad_circuit_to_fit(weber_circ, G_GOOGLE_WEBER)
        weber_circ = weber_sabre.run(weber_circ)
        weber_cx = weber_circ.count_ops()['cx']
        # Aspen9
        aspen9_circ = circ.copy()
        _pad_circuit_to_fit(aspen9_circ, G_RIGETTI_ASPEN9)
        aspen9_circ = aspen9_sabre.run(aspen9_circ)
        aspen9_cx = aspen9_circ.count_ops()['cx']
        # Put data in dict.
        data['qubit count'].append(n_qubits)
        data['circ original cnots'].append(og_cx)
        data['circ cnots after sabre, ibm toronto'].append(toronto_cx)
        data['circ cnots after sabre, google weber'].append(weber_cx)
        data['circ cnots after sabre, rigetti aspen9'].append(aspen9_cx)
        
        for x in data:
            print('%s: %.3f' % (x, data[x][-1]))
    return data
