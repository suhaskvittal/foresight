"""
    author: Suhas Vittal
    date:   2 November 2021
"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import *
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram as ph

from fs_exec import _pad_circuit_to_fit, draw, exec_sim
from fs_exec import total_variation_distance as TVD
from fs_foresight import ForeSight
from fs_util import *
from fs_benchmark_pass import prerouting_qiskitopt3, postrouting_qiskitopt3

import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import os
import re
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

def parse_log_file(log_file, output_file):
    reader = open(log_file, 'r')

    data = defaultdict(list)
    used_benchmarks = []

    line = reader.readline()
    while line != '':
        line = line.strip()
        circ_name_search = re.search('\[(.+?)\]', line)
        stat_search = re.search('(.+?):\s*(-?\d+?\.\d\d\d)', line)
        if circ_name_search is not None:
            used_benchmarks.append(circ_name_search.group(1))
        elif stat_search is not None:
            data[stat_search.group(1)].append(float(stat_search.group(2)))
        else:
            print(line)
        line = reader.readline()
    reader.close()
    df = pd.DataFrame(data=data, index=used_benchmarks)
    df.to_csv(output_file)

def plateau_analysis(output_file):
    benchmark_folder, benchmarks = G_BV_L 

    coupling_map = G_100GRID

    sabre = postrouting_qiskitopt3(coupling_map, routing_pass=SabreSwap(coupling_map, heuristic='decay'))
    foresight = postrouting_qiskitopt3(coupling_map, routing_pass=ForeSight(
        coupling_map,
        slack=3,
        solution_cap=16,
    ))
    layout_pass = prerouting_qiskitopt3(coupling_map)

    dataset = {}

    for qbfile in benchmarks:
        circuit = QuantumCircuit.from_qasm_file('%s/%s' % (benchmark_folder, qbfile))
        original_cnots = circuit.count_ops()['cx']
        sabre_array = [0]*100
        foresight_array = [0]*100
        for r in range(100):
            print('run ', r)
            mapped_circ = layout_pass.run(circuit)
            sabre_circ = sabre.run(mapped_circ)
            sabre_circ = transpile(
                sabre_circ,
                basis_gates=G_QISKIT_GATE_SET,
                layout_method='trivial',
                routing_method='none',
                optimization_level=0,
                approximation_degree=1,
            )
            sabre_cnots = sabre_circ.count_ops()['cx'] - original_cnots
            if r == 0:
                sabre_array[r] = sabre_cnots
            else:
                sabre_array[r] = min(sabre_cnots, sabre_array[r-1])
            print('sabre[%d] = ' % r, sabre_array[r])
            foresight_circ = foresight.run(mapped_circ)
            foresight_circ = transpile(
                foresight_circ,
                basis_gates=G_QISKIT_GATE_SET,
                layout_method='trivial',
                routing_method='none',
                optimization_level=3,
                approximation_degree=1,
            )
            foresight_cnots = foresight_circ.count_ops()['cx'] - original_cnots 
            if r == 0:
                foresight_array[r] = foresight_cnots    
            else:
                foresight_array[r] = min(foresight_cnots, foresight_array[r-1])
            print('foresight[%d] = ' % r, foresight_array[r])
        dataset[qbfile] = {
            'sabre runs': sabre_array,
            'foresight runs': foresight_array
        }
    writer = open(output_file, 'wb')
    pkl.dump(dataset, writer)
    writer.close()

def runtime_analysis(output_file):
    dataset = {}

    for solncap in [2,4,8,16,32]:
        for delta in [0,1,2,3,4,5]:
            df_time = pd.read_csv('data/raw/runtime_analysis/bvl_S=%d_D=%d.csv' % (solncap,delta))
            df_mem = pd.read_csv('data/raw/runtime_analysis/bvl_mem_S=%d_D=%d.csv' % (solncap,delta))

            header_index = 'Unnamed: 0'
            circuits = list(df_time.index.values)
            d = {
                'circuit details': {
                    df_time[header_index][x]: {
                        'original cnots': df_time['Original CNOTs'][x],
                        'qubits': df_time['qubits'][x]
                    } for x in circuits 
                },
                'foresight': {
                    df_time[header_index][x]: {
                        'cnots added': df_time['ForeSight-D CNOTs'][x],
                        'final depth': df_time['ForeSight-D Depth'][x],
                        'time': df_time['ForeSight-D Time'][x],
                        'memory': df_mem['ForeSight-D Memory'][x]
                    } for x in circuits
                }
            }
            dataset[(solncap,delta)] = d
    writer = open(output_file, 'wb')
    pkl.dump(dataset, writer)
    writer.close()

def _df_to_pydict(df, perf=False, mem=False, noisy=False, ssonly=False, tket=False, sim_counts=''):
    if sim_counts != '':
        with open(sim_counts, 'rb') as reader:
            counts = pkl.load(reader) 
    else:
        counts = {}
    d = {
            'original': {
                df['Unnamed: 0'][x]: {
                    'original cnots': df['Original CNOTs'][x]
                } for x in list(df.index.values)
            },
            'counts': counts,
            'sabre': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['SABRE CNOTs'][x],
                    'final depth': df['SABRE Depth'][x],
                    'execution time': df['SABRE Time'][x],
                    'memory': df['SABRE Memory'][x] if mem else 0.0,
                    'sabre tvd': df['SABRE TVD'][x] if noisy else 0.0
                } for x in list(df.index.values) 
            },
            'foresight': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['ForeSight-D CNOTs'][x],
                    'final depth': df['ForeSight-D Depth'][x],
                    'execution time': df['ForeSight-D Time'][x],
                    'memory': df['ForeSight-D Memory'][x] if mem else 0.0,
                    'ips tvd': df['ForeSight-D TVD'][x] if noisy else 0.0
                } for x in list(df.index.values) 
            },
            'noisy foresight': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['Noisy ForeSight-D CNOTs'][x],
                    'final depth': df['Noisy ForeSight-D Depth'][x],
                    'execution time': df['ForeSight-D Time'][x],
                    'noisy ips tvd': df['Noisy ForeSight-D TVD'][x],
                    'relative tvd to sabre': df['SABRE Relative TVD'][x],
                    'relative tvd to foresight': df['ForeSight-D Relative TVD'][x]
                } for x in list(df.index.values)
            } if noisy else {},
            'foresight (shallow solve only)': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['ForeSight SSOnly CNOTs'][x],
                    'final depth': df['ForeSight SSOnly Depth'][x],
                    'execution time': df['ForeSight SSOnly Time'][x],
                    'memory': df['ForeSight SSOnly Memory'][x] if mem else 0.0 
                } for x in list(df.index.values) 
            } if ssonly else {},
            'astar': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['A* CNOTs'][x],
                    'final depth': df['A* Depth'][x],
                    'execution time': df['A* Time'][x],
                    'memory': df['A* Memory'][x] if mem else 0.0
                } for x in list(df.index.values) 
            } if perf else {},
            'tket': {
                df['Unnamed: 0'][x]: {
                    'cnots added': df['TKET CNOTs'][x],
                    'final depth': df['TKET Depth'][x],
                    'execution time': df['TKET Time'][x],
                    'memory': df['TKET Memory'][x] if mem else 0.0
                } for x in list(df.index.values) if tket 
            }
        }
    return d

def get_swaps_dataset(pickle_file, excel_sub_type):
    dataset = {}

    for coupling_map in ['aspen', 'weber', 'tokyo']:
        df = pd.read_csv('data/raw/performance/with_qiskitopt3/%s_%s.csv' % (coupling_map, excel_sub_type))
        dataset[coupling_map] = _df_to_pydict(df, perf=True, mem=False, ssonly=False, tket=True)
    # pickle dataset
    with open(pickle_file, 'wb') as writer:
        pkl.dump(dataset, writer)

def get_path_sweep_dataset(pickle_file):
    dataset = {}
    for i in [1, 2, 4, 8, 16, 32]:
        df = pd.read_csv('data/raw/path-sweep/weber_path_sweep_zulehner_partial_%d.csv' % i)
        dataset[i] = _df_to_pydict(df)
    with open(pickle_file, 'wb') as writer:
        pkl.dump(dataset, writer)

def get_slack_sweep_dataset(pickle_file):
    dataset = {}
    for i in [0, 1, 2, 3, 4, 5]:
        df = pd.read_csv('data/raw/slack-sweep/weber_slack_sweep_zulehner_partial_%d.csv' % i)
        dataset[i] = _df_to_pydict(df)
    with open(pickle_file, 'wb') as writer:
        pkl.dump(dataset, writer)
    
def get_noise_sweep_dataset(pickle_file):
    dataset = {}
    with open('routed_qobjs/weber_circ.pkl', 'rb') as reader:
        d = pkl.load(reader)
        dataset['base'] = {
            x: {
                'counts': d[x]['counts'],
                'sabre': {
                    'circ': d[x]['sabre'],
                    'cx': d[x]['sabre'].count_ops()['cx']
                },
                'foresight': {
                    'circ': d[x]['foresight'],
                    'cx': d[x]['foresight'].count_ops()['cx']
                },
                'noisy foresight': {
                    'circ': d[x]['noisy foresight'],
                    'cx': d[x]['noisy foresight'].count_ops()['cx']
                }
            } for x in d
        }
    for i in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        with open('routed_qobjs/weber_circ_sweep_%.8f.pkl' % i, 'rb') as reader:
            dataset[i] = pkl.load(reader)
    with open(pickle_file, 'wb') as writer:
        pkl.dump(dataset, writer)

def get_bv_dataset(pickle_file):
    dataset = {}
    for i in ['large', 'vlarge']:
        df = pd.read_excel('data/bv_%s.xlsx' % i)
        dataset[i] = _df_to_pydict(df, mem=True)
    with open(pickle_file, 'wb') as writer:
        pkl.dump(dataset, writer)
