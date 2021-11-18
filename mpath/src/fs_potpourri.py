"""
    author: Suhas Vittal
    date:   2 November 2021
"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import *
from qiskit.transpiler import CouplingMap, PassManager

from fs_exec import _pad_circuit_to_fit, draw
from fs_foresight import ForeSight
from fs_util import *

import pandas as pd
import pickle as pkl

import os
from collections import defaultdict

def convert_integer_to_bstring(num):
	'''
	Function to convert an integer into bitstring
	'''
	bstring = ''
	flag = 0 
	if(num>1):
		bstring = convert_integer_to_bstring(num // 2)
	bstring = bstring+ str(num % 2)
	return bstring

def padding_for_binary(bstring, expected_length):
	''' 
	Function to pad a bitstring with 0s and stretch it to a given length
	'''
	curr_length = len(bstring)
	if(expected_length > curr_length):
		diff_length = expected_length - curr_length
		padding = ''
		for i in range(diff_length):
			padding = padding + str(0)
		bstring = padding + bstring
	return bstring

def get_key_from_decimal(num,length):
	''' 
	Function to convert a decimal to a key of a given length
	'''
	bstr = convert_integer_to_bstring(num)
	key = padding_for_binary(bstr, length)
	return key

def generate_random_bv_circuit(num_qregs,hidden_key,fname):
	'''
	Function to create a random bv circuit for a given input size and hidden bitstring
	'''
#	workload_dir = './workloads/bv/'
#	if not os.path.exists(workload_dir):
#		os.makedirs(workload_dir)
#	fname = workload_dir + 'bv'+str(num_qregs)+'_hidden_key_'+str(hidden_key)+'.qasm'
	f = open(fname,"w+")
	# dump the qreg and creg information
	include_str = 'OPENQASM 2.0;\n'
	f.write(include_str)
	include_str = 'include "qelib1.inc";\n'
	f.write(include_str)
	include_str = 'qreg q[' + str(num_qregs) + '];\n'
	f.write(include_str)
	include_str = 'creg c[' + str(num_qregs-1) + '];\n'
	f.write(include_str)
	# generate the h gates 
	for i in range(num_qregs-1):
		h_str = 'h q[' + str(i) + '];\n'
		f.write(h_str)
	# take the ancilla qubit to |-> state
	ancilla_str = 'x q[' + str(num_qregs-1) + '];\n'
	f.write(ancilla_str)
	ancilla_str = 'h q[' + str(num_qregs-1) + '];\n'
	f.write(ancilla_str)
	# generate the bitstring for the given key
	hidden_bitstring = get_key_from_decimal(num=hidden_key,length= num_qregs-1)
	#print(hidden_bitstring)
	# generate the cnot gates
	for c in range(len(hidden_bitstring)):
		if(hidden_bitstring[c] == '1'):
			cx_str = 'cx q[' + str(c) + '], q[' + str(num_qregs-1) + '];\n'   
			f.write(cx_str) #print(cx_str)
	for i in range(num_qregs-1):
		h_str = 'h q[' + str(i) + '];\n'
		f.write(h_str)
	# dump the measurement operations
	for i in range(num_qregs-1):
		meas_str = 'measure q[' + str(i) + '] -> c[' + str(i) + '];\n'
		f.write(meas_str)
	f.close()

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

def get_dataset(pickle_file, excel_sub_type):
    dataset = {}

    for coupling_map in ['toronto','aspen9', 'weber', 'tokyo']:
        df = pd.read_excel('data/%s_%s.xlsx' % (coupling_map, excel_sub_type))
        dataset[coupling_map] = {
            'original': {
                df['Unnamed: 0'][x]: {
                    'original cnots': df['Original CNOTs'][x]
                } for x in list(df.index.values)
            },
            'sabre': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['SABRE CNOTs'][x],
                    'final depth': df['SABRE Depth'][x],
                    'execution time': df['SABRE Time'][x],
                    'memory': df['SABRE Memory'][x]
                } for x in list(df.index.values) 
            },
            'ips': {  # IPS is legacy name, used to not break code
                df['Unnamed: 0'][x]: {
                    'cnots added': df['ForeSight CNOTs'][x],
                    'final depth': df['ForeSight Depth'][x],
                    'execution time': df['ForeSight Time'][x],
                    'memory': df['ForeSight Memory'][x]
                } for x in list(df.index.values) 
            },
            'ips (shallow solve only)': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['ForeSight SSOnly CNOTs'][x],
                    'final depth': df['ForeSight SSOnly Depth'][x],
                    'execution time': df['ForeSight SSOnly Time'][x],
                    'memory': df['ForeSight SSOnly Memory'][x]
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
                    'execution time': df['A* Time'][x],
                    'memory': df['A* Memory'][x]
                } for x in list(df.index.values) 
            }
        }
    # pickle dataset
    with open(pickle_file, 'wb') as writer:
        pkl.dump(dataset, writer)

