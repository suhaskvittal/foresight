"""
    author: Suhas Vittal
    date:   2 November 2021
"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import *
from qiskit.transpiler import CouplingMap, PassManager

from fs_exec import _pad_circuit_to_fit, draw
from fs_ips import ForeSight
from fs_util import G_QAOA,\
                    G_IBM_TORONTO,\
                    G_RIGETTI_ASPEN9,\
                    G_GOOGLE_WEBER,\
                    G_QISKIT_GATE_SET

import pandas as pd
import pickle as pkl

from collections import defaultdict

def figure1_circ(filename):
    cmap = CouplingMap.from_ring(6)
    circ = QuantumCircuit.from_qasm_file(filename)
    sabre = PassManager([TrivialLayout(cmap), ApplyLayout(), SabreSwap(cmap, heuristic='basic')])
    ips = PassManager([TrivialLayout(cmap), ApplyLayout(), ForeSight(cmap, slack=3, solution_cap=1)])
    sabre_circ = sabre.run(circ)
    ips_circ = ips.run(circ)
    draw(circ)
    draw(sabre_circ)
    draw(ips_circ)
    print(sabre_circ.count_ops())
    print(ips_circ.count_ops())
    return circ, sabre_circ, ips_circ

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

def get_dataset1(pickle_file):
    dataset = {}

    for coupling_map in ['toronto', 'aspen9', 'weber', 'tokyo']:
        df = pd.read_excel('data/sabre_initial/%s_zulehner.xlsx' % coupling_map)
        dataset[coupling_map] = {
            'sabre': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['SABRE CNOTs'][x],
                    'final depth': df['SABRE Depth'][x],
                    'execution time': df['SABRE Time'][x]
                } for x in list(df.index.values) 
            },
            'foresight': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['ForeSight CNOTs'][x],
                    'final depth': df['ForeSight Depth'][x],
                    'execution time': df['ForeSight Time'][x]
                } for x in list(df.index.values) 
            },
            'foresight (shallow solve only)': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['ForeSight SSOnly CNOTs'][x],
                    'final depth': df['ForeSight SSOnly Depth'][x],
                    'execution time': df['ForeSight SSOnly Time'][x]
                } for x in list(df.index.values) 
            },
            'best of sabre and ips': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['Best CNOTs'][x],
                    'final depth': df['Best Depth'][x]
                } for x in list(df.index.values)
            },
            'astar': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['A* CNOTs'][x],
                    'final depth': df['A* Depth'][x],
                    'execution time': df['A* Time'][x]
                } for x in list(df.index.values) 
            }
        }
    # pickle dataset
    with open(pickle_file, 'wb') as writer:
        pkl.dump(dataset, writer)

