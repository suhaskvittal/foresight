import os
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ
## packages required for qaoa
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.random_graphs import random_regular_graph
#from qiskit.providers.ibmq import least_busy
#from qiskit.tools.monitor import job_monitor
#from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import IBMQ
#IBMQ.update_account() : was only relevant for Qiskit update
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()
import qiskit
import datetime
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import numpy as np
import math
from qiskit.providers.aer.noise.errors import standard_errors as SE
from qiskit.providers.aer.noise.device import models 
IBMQ.save_account('')	### FIXME: insert token here from ibm q account
#print("Available backends:")
provider = IBMQ.get_provider(hub='ibm-q')
#print(provider.backends())
from qiskit.qasm import Qasm
# useful additional packages 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
#from qiskit.aqua import run_algorithm
#from qiskit.aqua.input import EnergyInput
from qiskit.optimization.applications.ising import max_cut, tsp, common
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
''' overall helper functions '''
def write_qasm_file_from_qobj(output_file,qobj):
	''' 
	Function to write a Quantum Object into a given output file QASM
	'''
	f= open(output_file,"w+")
	f.write(qobj.qasm())
	f.close()
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


def generate_random_ghz_circuit(num_qregs):
	'''
	Function to create a random bv circuit for a given input size
	'''

	workload_dir = './workloads/ghz/'
	if not os.path.exists(workload_dir):
		os.makedirs(workload_dir)
	fname = workload_dir + 'ghz'+str(num_qregs)+'.qasm'
	f = open(fname,"w+")
	# dump the qreg and creg information
	include_str = 'OPENQASM 2.0;\n'
	f.write(include_str)
	include_str = 'include "qelib1.inc";\n'
	f.write(include_str)
	include_str = 'qreg q[' + str(num_qregs) + '];\n'
	f.write(include_str)
	include_str = 'creg c[' + str(num_qregs) + '];\n'
	f.write(include_str)
	# generate the h gate
	h_str = 'h q[0];\n'
	f.write(h_str)
	# generate the cnot gates
	for i in range(num_qregs-1):
		cx_str = 'cx q[' + str(i) + '], q['+str(i+1)+'];\n'
		f.write(cx_str)
	# dump the measurement operations
	for i in range(num_qregs):
		meas_str = 'measure q[' + str(i) + '] -> c[' + str(i) + '];\n'
		f.write(meas_str)
	f.close()

''' Functions for generating the QFT circuit '''
def qft_rotations(circuit, n):
	"""Performs qft on the first n qubits in circuit (without swaps)"""
	if n == 0:
		return circuit
	n -= 1
	circuit.h(n)
	for qubit in range(n):
		circuit.cu1(pi/2**(n-qubit), qubit, n)
	# At the end of our function, we call the same function again on
	# the next qubits (we reduced n by one earlier in the function)
	qft_rotations(circuit, n)

def swap_registers(circuit, n):
	for qubit in range(n//2):
		circuit.swap(qubit, n-qubit-1)
	return circuit

def qft(circuit, n):
	"""QFT on the first n qubits in circuit"""
	qft_rotations(circuit, n)
	swap_registers(circuit, n)
	return circuit

def generate_qft_circuit(num_qregs,encoding_value):
	'''
	Function to generate a QFT circuit with a given number of qubits and encoding value where the encoding value ranges from 0 to 2^n-1
	'''
	workload_dir = './workloads/qft/'
	if not os.path.exists(workload_dir):
		os.makedirs(workload_dir)
	fname = workload_dir + 'qft'+str(num_qregs)+'_encoding_value_'+str(encoding_value)+'.qasm'
	qc = QuantumCircuit(num_qregs,num_qregs)
	## encoding phase
	bstr = get_key_from_decimal(num=encoding_value,length=num_qregs)
	for c in range(num_qregs):
		if(bstr[c]=='1'):
			qc.x(c)
	qc = qft(qc,num_qregs)
	mqs = range(num_qregs)
	qc.measure(mqs, mqs)
	write_qasm_file_from_qobj(output_file=fname,qobj=qc)

'''Functions for inverse qft'''
def inverse_qft(circuit, n):
	"""Does the inverse QFT on the first n qubits in circuit"""
	# First we create a QFT circuit of the correct size:
	qft_circ = qft(QuantumCircuit(n), n)
	# Then we take the inverse of this circuit
	invqft_circ = qft_circ.inverse()
	# And add it to the first n qubits in our existing circuit
	circuit.append(invqft_circ, circuit.qubits[:n])
	swap_registers(circuit, n)
	return circuit.decompose() # .decompose() allows us to see the individual gates

def generate_inverse_qft_circuit(num_qregs,encoding_value):
	'''
	Function to generate a inverse QFT circuit with a given number of qubits and encoding value where the encoding value ranges from 0 to 2^n-1
	'''
	workload_dir = './workloads/inv_qft/'
	if not os.path.exists(workload_dir):
		os.makedirs(workload_dir)
	fname = workload_dir + 'inverse_qft'+str(num_qregs)+'_encoding_value_'+str(encoding_value)+'.qasm'
	qc = QuantumCircuit(num_qregs,num_qregs)
	for c in range(num_qregs):
		qc.h(c)
	qidx = num_qregs-1
	for i in range(num_qregs):
		power = math.pow(2,i)
		qc.u1(encoding_value*pi/power,qidx)
		qidx = qidx-1
	qc = inverse_qft(qc,num_qregs)
	mqs = range(num_qregs)
	qc.measure(mqs, mqs)
	write_qasm_file_from_qobj(output_file=fname,qobj=qc)

''' graph generator functions and qaoa circuit generation'''
def create_random_graph(num_qregs,degree):
	g = erdos_renyi_graph(num_qregs,degree)
	edge_list = []
	for edge in g.edges():
		list_edge = list(edge)
		## for now every edge has weight 1.0
		list_edge.append(1.0)
		list_edge = tuple(list_edge)
		edge_list.append(list_edge)
	return edge_list

def generate_graph(n,E):
	V = np.arange(0,n,1)
	G = nx.Graph()
	G.add_nodes_from(V)
	G.add_weighted_edges_from(E)
	return G

def generate_weight_matrix(n,G):
	w = np.zeros([n,n])
	for i in range(n):
		for j in range(n):
			temp = G.get_edge_data(i,j,default=0)
			if temp != 0:
				w[i,j] = temp['weight'] 
		        
	return w

def get_qaoa_circuit(wts,G,p,entanglement_type,output_file):
	qubitOp, offset = max_cut.get_operator(wts)
	seed = 10598
	spsa = SPSA(max_trials=300)
	ry = RY(qubitOp.num_qubits, depth=p, entanglement=entanglement_type)
	vqe = VQE(qubitOp, ry, spsa)
	backend = Aer.get_backend('qasm_simulator')
	quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
	result = vqe.run(quantum_instance)
	"""declarative approach
	algorithm_cfg = {
	    'name': 'VQE'
	}
	
	optimizer_cfg = {
	    'name': 'SPSA',
	    'max_trials': 300
	}
	
	var_form_cfg = {
	    'name': 'RY',
	    'depth': 5,
	    'entanglement': 'linear'
	}
	
	params = {
	    'problem': {'name': 'ising', 'random_seed': seed},
	    'algorithm': algorithm_cfg,
	    'optimizer': optimizer_cfg,
	    'variational_form': var_form_cfg,
	    'backend': {provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'}
	}
	
	result = run_algorithm(params, algo_input)
	"""
	pos = nx.spring_layout(G)
	x = common.sample_most_likely(result['eigvecs'][0])
	#print('energy:', result['energy'])
	#print('time:', result['eval_time'])
	#print('max-cut objective:', result['energy'] + offset)
	#print('solution:', max_cut.get_graph_solution(x))
	#print('solution objective:', max_cut.max_cut_value(x, wts))
	
	#colors = ['r' if max_cut.get_graph_solution(x)[i] == 0 else 'b' for i in range(n)]
	#nx.draw_networkx(G, node_color=colors, node_size=600, alpha = .8, pos=pos)
	qc= vqe.get_optimal_circuit()
	c = ClassicalRegister(vqe._operator.num_qubits, name='c')
	qc.add_register(c)
	q = qc.qubits
	qc.measure(q, c)
	#print(qc.qasm())
	write_qasm_file_from_qobj(output_file,qc)

def remove_barriers_from_circuit(readfilename,writefilename):
	wf = open(writefilename, "w+")
	with open(readfilename) as rf:
		for line in rf:
			if 'barrier' not in line:
				wf.write(line)
	rf.close()
	wf.close()

def generate_qaoa_circuit(num_qregs,degree,depth,entanglement_type):
	'''Function to generate a random qaoa circuit
	Parameters: num_qregs: size of circ, degree= degree of connectivity in the random graph to be generated, 
				depth= qaoa algorithm parameter (theoretically ranges between 1 to n but typically <=5/6, 
				entanglement type= linear/full: linear creates a chain of cnots, full entanglement requires more connectivity
	'''
	workload_dir = './workloads/qaoa/'
	if not os.path.exists(workload_dir):
		os.makedirs(workload_dir)
	fname = workload_dir + 'qaoa'+str(num_qregs)+'_degree_'+str(degree)+'_depth_'+str(depth)+'_'+str(entanglement_type)+'.qasm'

	## create an arbitrary graph
	E = create_random_graph(num_qregs,degree)
	G = generate_graph(num_qregs,E)
	wts = generate_weight_matrix(num_qregs,G)
	temp_fname='temp.qasm' ## first dump the circuit and then remove any barriers present (since they are only logical)
	get_qaoa_circuit(wts,G,depth,entanglement_type,temp_fname)
	remove_barriers_from_circuit(temp_fname,fname)



#generate_qaoa_circuit(num_qregs=5,degree=0.9,depth=2,entanglement_type='full')
#generate_qft_circuit(num_qregs=4,encoding_value=3)
#generate_inverse_qft_circuit(num_qregs=4,encoding_value=3)
#generate_random_bv_circuit(num_qregs=4,hidden_key=2)
#generate_random_ghz_circuit(num_qregs=4)

## possible wrappers for sweeping
def Bernstein_Vazirani_Wrapper(num_qregs):
	state_space = int(math.pow(2,(num_qregs-1)))
	for hidden_key in range(state_space):
		generate_random_bv_circuit(num_qregs=num_qregs,hidden_key=hidden_key)

def QFT_Wrapper(num_qregs):
	state_space = int(math.pow(2,num_qregs))
	for encoding_value in range(state_space):
		generate_qft_circuit(num_qregs=num_qregs,encoding_value=encoding_value)

def Inverse_QFT_Wrapper(num_qregs):
	state_space = int(math.pow(2,num_qregs))
	for encoding_value in range(state_space):
		generate_inverse_qft_circuit(num_qregs=num_qregs,encoding_value=encoding_value)

#Bernstein_Vazirani_Wrapper(num_qregs=3)
#QFT_Wrapper(num_qregs=3)
#Inverse_QFT_Wrapper(num_qregs=3)
