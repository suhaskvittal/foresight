"""
    author: Suhas Vittal
    date:   20 March 2022
"""

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import *
from qiskit.transpiler import PassManager

from qiskit.quantum_info import Operator
from qiskit.extensions import UnitaryGate

from fs_util import read_arch_file
from fs_foresight import *

import numpy as np
import math

def google_sycamore_noise_model(backend, noise_file, noise_factor=1.0):
    sycamore = backend
    # Noise parameters from Weber (some estimated)
    G1Q_TIME = 25  # ns
    CX_TIME = 32
    MEAS_TIME = 4000
    T1 = 15000
    T2 = T1*0.5
    # Read noise file.
    reader = open(noise_file, 'r')
    num_qubits = int(reader.readline())
    g_2q_err = {'cx':{}}
    g_2q_time = {'cx':{}}
    g_1q_err = {g:[0 for _ in range(num_qubits)] for g in ['u1', 'u2', 'u3']}
    g_1q_time = {g:[0 for _ in range(num_qubits)] for g in ['u1', 'u2', 'u3']}
    prob_m1_g0 = [0 for _ in range(num_qubits)]
    prob_m0_g1 = [0 for _ in range(num_qubits)]
    meas_time = [0 for _ in range(num_qubits)]
    coh_t1 = [0 for _ in range(num_qubits)]
    coh_t2 = [0 for _ in range(num_qubits)]
    for _ in range(num_qubits):
        split_line = reader.readline().split(' ')
        q, e_1q, pm1g0, pm0g1 = int(split_line[0]), float(split_line[1]),\
                float(split_line[2]), float(split_line[3])
        for g in g_1q_err:
            g_1q_err[g][q] = e_1q*noise_factor
            g_1q_time[g][q] = G1Q_TIME
        prob_m1_g0[q] = pm1g0*noise_factor
        prob_m0_g1[q] = pm0g1*noise_factor
        meas_time[q] = MEAS_TIME
        coh_t1[q] = T1
        coh_t2[q] = T2
    line = reader.readline()  # now read until the end of the file
    while line != '':
        split_line = line.split(' ')
        q1, q2, e_2q = int(split_line[0]), int(split_line[1]), float(split_line[2])
        g_2q_err['cx'][(q1,q2)] = e_2q*noise_factor
        g_2q_err['cx'][(q2,q1)] = e_2q*noise_factor
        g_2q_time['cx'][(q1,q2)] = CX_TIME
        g_2q_time['cx'][(q2,q1)] = CX_TIME
        line = reader.readline()
    noise_model = _build_noise_model(
        num_qubits,
        sycamore,
        g_2q_err,
        g_2q_time,
        g_1q_err,
        g_1q_time,
        prob_m1_g0,
        prob_m0_g1,
        meas_time,
        coh_t1,
        coh_t2,
        warnings=False,
        use_1qerror=True,
        use_readout=True,
        use_coherence=False
    )
    cx_error_rates = {}
    for (i,j) in sycamore.get_edges():
        if (i,j) in g_2q_err['cx']:
            e = g_2q_err['cx'][(i,j)]
        else:
            e = g_2q_err['cx'][(j,i)]
        cx_error_rates[(i,j)] = e
    # Build vertex weights and readout weights for ForeSight
    sq_error_rates = {}
    ro_error_rates = {}
    for i in range(num_qubits):
        mean_1qe = np.mean([
            g_1q_err[g][i] for g in g_1q_err
        ])
        mean_mse = 0.5*(prob_m1_g0[i] + prob_m0_g1[i])
        sq_error_rates[i] = mean_1qe
        ro_error_rates[i] = mean_mse
    return noise_model, cx_error_rates, sq_error_rates, ro_error_rates

def _build_noise_model(
    num_qubits,
    coupling_map,
    g_2q_err,
    g_2q_time,
    g_1q_err,
    g_1q_time,
    prob_m1_g0,
    prob_m0_g1,
    meas_time,
    coh_t1,
    coh_t2,
    warnings=False,
    use_2qerror=True,
    use_1qerror=True,
    use_readout=True,
    use_coherence=True
):
    noise_model = NoiseModel()
    if use_1qerror:
        # Add single qubit error
        for i in range(num_qubits):
            for gate in g_1q_err:
                # Gate error
                e = depolarizing_error(g_1q_err[gate][i], 1)
                noise_model.add_quantum_error(e, [gate], [i], warnings=warnings)
                if use_coherence:
                    # Coherence error
                    e = thermal_relaxation_error(coh_t1[i], coh_t2[i], g_1q_time[gate][i])
                    noise_model.add_quantum_error(e, [gate], [i], warnings=warnings)
    if use_2qerror:
        # Add double qubit error
        for (i, j) in coupling_map.get_edges():
            for gate in g_2q_err:
                # Gate error
                e = depolarizing_error(g_2q_err[gate][(i,j)], 2)
                noise_model.add_quantum_error(e, [gate], [i,j], warnings=warnings)
                if use_coherence:
                    # Coherence error
                    e = thermal_relaxation_error(coh_t1[i], coh_t2[i], g_2q_time[gate][(i,j)])
                    e = e.expand(
                        thermal_relaxation_error(coh_t1[j], coh_t2[j], g_2q_time[gate][(i,j)]))
                    noise_model.add_quantum_error(e, [gate], [i,j], warnings=warnings)
    if use_readout:
        # Add measurement errors
        for i in range(num_qubits):
            # Gate error
            p10 = _truncate(prob_m1_g0[i], 3)  # P(measure 1 given 0)
            p01 = _truncate(prob_m0_g1[i], 3)  # P(measure 0 given 1)
            e = ReadoutError([[1 - p10, p10], [p01, 1 - p01]])
            noise_model.add_readout_error(e, [i], warnings=warnings)
            if use_coherence:
                # Coherence error
                e = thermal_relaxation_error(coh_t1[i], coh_t2[i], meas_time[i])
                noise_model.add_quantum_error(e, ['measure'], [i], warnings=warnings)
    return noise_model

def _truncate(n, decimals=0):
    f = 10**decimals
    return np.round(n*f) / f
   
# Evaluation metrics

def normalize_dict(input_dict):
    epsilon = 0.0000001
    if sum(input_dict.values()) == 0:
        ##print(input_dict)
        ##print('Error, dictionary with total zero elements!!')    
        for k,v in input_dict.items():
            input_dict[k] = epsilon
    factor=1.0/sum(input_dict.values())
    #if(factor == 0):
    #    print(factor,sum(input_dict.values())) 
    for k in input_dict:
        input_dict[k] = input_dict[k]*factor

    for k,v in input_dict.items():
        if(v==1):
            input_dict[k] = 1-epsilon

    return input_dict

## adding my own functions on tpop of swamit
def compute_pst(ideal_histogram,noisy_histogram,num_sols):
    ''' 
    Function to compute PST
    '''
    # determine the total best solutions from ideal
    sorted_histogram = sorted(ideal_histogram.items(), key=lambda x: x[1], reverse=True)
    successful_trials_counter = 0 
    for i in range(num_sols):
        search_key = sorted_histogram[i][0]
        for key,value in noisy_histogram.items():
            if(key == search_key):
                successful_trials_counter = successful_trials_counter + value
    # compute PST
    total_trials = sum(noisy_histogram.values())
    if(successful_trials_counter <=1.0): #already a pdf
        pst = successful_trials_counter
    else:
        pst = successful_trials_counter/total_trials
    return pst 

def compute_ist(ideal_histogram,noisy_histogram,num_sols):
    ''' 
    Function to compute IST
    '''
    # determine the total best solutions from ideal
    sorted_histogram = sorted(ideal_histogram.items(), key=lambda x: x[1], reverse=True)
    # sort the noisy histogram
    sorted_noisy_histogram = sorted(noisy_histogram.items(), key=lambda x: x[1], reverse=True)
    # probability of correct answer
    successful_trials_counter = 0 
    for i in range(num_sols):
        search_key = sorted_histogram[i][0]
        for key,value in noisy_histogram.items():
            if(key == search_key):
                successful_trials_counter = successful_trials_counter + value
    # get the solution keys
    solution_keys = []
    for j in range(num_sols):
        solution_keys.append(sorted_histogram[j][0])
        
    error_counter = 0 
    for i in range(len(sorted_noisy_histogram)):
        search_key = sorted_noisy_histogram[i][0]
        if search_key not in solution_keys:
            error_counter = sorted_noisy_histogram[i][1]
            break
    if error_counter == 0:
        ist = math.inf
    else:
        ist = successful_trials_counter/error_counter

    return ist 

def pst_update_dist(dict1, dict2, ideal_counts_vector):
    
    Counter1 = Counter(dict1)
    Counter2 = Counter(dict2)
     
    
    pst = compute_pst(ideal_counts_vector[0],dict2,1)
    for key in Counter2.keys():
        value = Counter2[key]
        value = value * pst
        Counter2[key] = value

    FinalCounter = Counter1 + Counter2
    final_dict = dict(FinalCounter)
    return final_dict
    
def sum_of_a_power(dict1):
    Counter1 = Counter(dict1)
    result = 0
    
    for key, value in Counter1.items():
        result += value ** 2.5
        
    return result

def fancy_update_dist(dict1, dict2):
    
    Counter1 = Counter(dict1)
    Counter2 = Counter(dict2)
    
    #Create uniform dict for hdist
    uniform_counter = Counter(dict2)
    for key, value in uniform_counter.items():
        uniform_counter[key] = 1/len(Counter2)
    uniform_dict = dict(uniform_counter)
    hdist,corr = compute_hdist(dict2,uniform_dict)
    print(hdist)
    
    #size_diff = sum_of_squares(dict2)
    size_diff = hdist
    size_diff = sum_of_a_power(dict2)

    for key, value in Counter2.items():
        value = value * size_diff
        Counter2[key] = value

    FinalCounter = Counter1
    scalar = -0.5
    for key, value in Counter2.items():
        old_value = (FinalCounter[key] + value)        
        FinalCounter[key] = old_value
        #print(old_value, " ", new_value)
    
    final_dict = dict(FinalCounter)
    return final_dict

def update_dist(dict1,dict2):
    ''' 
    Function to merge two dictionaries in to a third one
    '''
    dict3 = Counter(dict1) + Counter(dict2) 
    dict3 = dict(dict3)
    return dict3
def truncate(number, decimals=0):
    """ 
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor
    
def removekey(d, key_list):
    for i in key_list:
        r = dict(d)
        del r[i]
    
    return r
    
def compute_hdist(dist_a,dist_b):
	_in1 = normalize_dict(dist_a.copy())
	_in2 = normalize_dict(dist_b.copy())

	epsilon = 0.00000001
	# update the dictionaries

	for key in _in1.keys():
		if key not in _in2:
			_in2[key] = epsilon # add new entry
	
	for key in _in2.keys():
		if key not in _in1:
			_in1[key] = epsilon # add new entry
	
	# both dictionaries should have the same keys by now
	if set(_in1.keys()) != set(_in2.keys()):
		print('Error : dictionaries need to be re-adjusted')

	## normalize the dictionaries

	_in1 = normalize_dict(_in1)
	_in2 = normalize_dict(_in2)
	
	#print(_in1)
	#print(_in2)

	list_of_squares = []
	for key,p in _in1.items():
		for _key,q in _in2.items():
			if key == _key:
				s = (math.sqrt(p) - math.sqrt(q)) ** 2
				list_of_squares.append(s)
				break
	# calculate the sum of squares
	sosq = sum(list_of_squares)
	hdist = math.sqrt(sosq)/math.sqrt(2)
	corr = 1-hdist	
	return hdist,corr
				
## functions to evaluate the expectation value


def compute_weight_matrix(_G):
    n = len(_G.nodes())
    w = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            temp = _G.get_edge_data(i,j,default=0)
            if temp != 0:
                w[i,j] = temp['weight']
    return w

def compute_cost_of_cut(graph_cut,weight_matrix):
    
    n=len(graph_cut)
    cost = 0
    for i in range(n):
        for j in range(n):
            cost = cost + weight_matrix[i,j]* int(graph_cut[i])* (1- int(graph_cut[j]))
    
    return cost


def compute_expected_value(_out_dict,in_graph):
    
    # check if cut is valid
    
    out_dict = normalize_dict(_out_dict.copy())
#     print(out_dict)
    W = compute_weight_matrix(in_graph)
    E = 0
    for key in out_dict:
        key_lst=[] 
        key_lst[:0]=key 
        cost = compute_cost_of_cut(key_lst,W)
        E += out_dict[key]*cost
    
    return E

def obtain_approximation_ratio(_out_dict,in_graph,solution):
	## obtain value of cost function from mean of all samples
	#print(solution)
	W = compute_weight_matrix(in_graph)
	mean_from_all_samples = compute_expected_value(_out_dict,in_graph)
	best_cut_value = compute_cost_of_cut(solution,W)
	#print(mean_from_all_samples,best_cut_value)
	
	return mean_from_all_samples/best_cut_value

def obtain_approximation_ratio_gap(ideal_dict,noisy_dict,in_graph,solution):
	ar_ideal = obtain_approximation_ratio(ideal_dict,in_graph,solution)
	ar_noisy = obtain_approximation_ratio(noisy_dict,in_graph,solution)

	return 100*abs(ar_ideal-ar_noisy)/ar_ideal 

def norm(numbers):
    if isinstance(numbers,list)==1:
        sum_of_numbers = 0 
        for i in numbers:
            sum_of_numbers = sum_of_numbers + math.pow(i,2)
        return math.sqrt(sum_of_numbers)
    else:
        return math.sqrt(math.pow(numbers,2))

def tvd_two_dist(p,q):
    _p = p.copy()
    _p = normalize_dict(_p)
    _q = q.copy()
    _q = normalize_dict(_q)
    
    epsilon = 0.0000000001
    ## match both dictionaries
    for key in _p.keys():
        if key not in _q.keys():
            _q[key] = epsilon
    
    for key in _q.keys():
        if key not in _p.keys():
            _p[key] = epsilon

    _p = normalize_dict(_p)
    _q = normalize_dict(_q)

    _q_rearranged = {}
    for key,value in _p.items():
        _q_rearranged[key] = _q[key]

    ## compute_tvd
    tvd = 0 
    for key,value in _p.items():
        diff = value - _q_rearranged[key]
        tvd = tvd + norm(diff)
    return tvd/2

def fidelity_from_tvd(p,q):
    epsilon = 0.0000001
    tvd = tvd_two_dist(p,q)
    fidelity = 1-tvd
    return fidelity

def root_mean_square_error(dist_a,dist_b):
    _in1 = normalize_dict(dist_a.copy())
    _in2 = normalize_dict(dist_b.copy())

    epsilon = 0.00000001
    # update the dictionaries

    for key in _in1.keys():
        if key not in _in2:
            _in2[key] = epsilon # add new entry

    for key in _in2.keys():
        if key not in _in1:
            _in1[key] = epsilon # add new entry

    # both dictionaries should have the same keys by now
    if set(_in1.keys()) != set(_in2.keys()):
        print('Error : dictionaries need to be re-adjusted')

    ## normalize the dictionaries

    _in1 = normalize_dict(_in1)
    _in2 = normalize_dict(_in2)

    #print(_in1)
    #print(_in2)


    list_of_squares = []
    for key,p in _in1.items():
        for _key,q in _in2.items():
            if key == _key:
                s = (p-q)**2
                list_of_squares.append(s)
                break
    # calculate the sum of squares
    root_mean_square_error = math.sqrt(sum(list_of_squares))/len(list_of_squares)
    return root_mean_square_error

def get_evaluation_output(ideal_dist,real_dist,metric='fidelity'):
    if metric == 'fidelity':
        eval_output = fidelity_from_tvd(ideal_dist,real_dist)
    elif metric == 'tvd':
        eval_output = tvd_two_dist(ideal_dist,real_dist)
    elif metric == 'pst':
        eval_output = compute_pst(ideal_dist,real_dist,1)
    elif metric == 'ist':
        eval_output = compute_ist(ideal_dist,real_dist,1)
    elif metric == 'hdist':
        eval_output,_ = compute_hdist(ideal_dist,real_dist)
    elif metric == 'corr':
        _, eval_output = compute_hdist(ideal_dist,real_dist)
    elif metric == 'rmse':
        eval_output = root_mean_square_error(ideal_dist,real_dist)
    return eval_output


if __name__ == '__main__':
    from sys import argv

    circ_file = argv[1]

    arch_file = '../arch/google_weber.arch'
    backend = read_arch_file(arch_file)
    noise_file = '../arch/noisy/google_weber.noise'

    _, cx_error_rates, sq_error_rates, ro_error_rates =\
        google_sycamore_noise_model(backend, noise_file)
    base_flags = FLAG_DEBUG | FLAG_ALAP
    compiler_noise_unaware = ForeSight(
        backend,
        slack=2,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        flags=base_flags
    )
    compiler_noise_aware = ForeSight(
        backend,
        slack=0.01,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        flags=base_flags | FLAG_NOISE_AWARE
    )

    fs1 = PassManager([TrivialLayout(backend), ApplyLayout(), compiler_noise_unaware])
    fs2 = PassManager([TrivialLayout(backend), ApplyLayout(), compiler_noise_aware])
    
    circ = QuantumCircuit.from_qasm_file(circ_file)
    print('=========NOISE UNAWARE==========')
    fs1.run(circ)
    print('=========NOISE AWARE==========')
    fs2.run(circ)

