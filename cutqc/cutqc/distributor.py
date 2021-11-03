import itertools
import numpy as np
from time import time
import argparse
import pickle
import os
import subprocess
import glob
import copy

from cutqc.helper_fun import get_dirname
from cutqc.post_process import get_combinations, build
from qiskit_helper_functions.non_ibmq_functions import read_dict, find_process_jobs

def distribute_load(total_load,capacities):
    assert total_load<=sum(capacities)
    loads = [0 for x in capacities]
    for slot_idx, load in reversed(list(enumerate(loads))):
        loads[slot_idx] = int(capacities[slot_idx]/sum(capacities)*total_load)
    total_load -= sum(loads)
    for slot_idx, load in reversed(list(enumerate(loads))):
        while total_load>0 and loads[slot_idx]<capacities[slot_idx]:
            loads[slot_idx] += 1
            total_load -= 1
    # print('distributed loads:',loads)
    assert total_load==0
    return loads

def initialize_dynamic_definition_schedule(counter,recursion_qubit,verbose):
    '''
    schedule[recursion_layer] =  {'smart_order','subcircuit_state','upper_bin'}
    subcircuit_state[subcircuit_idx] = ['0','1','active','merged']
    '''
    if verbose:
        print('Initializing first DD recursion, recursion_qubit=%d.'%recursion_qubit,flush=True)
    # print(counter)
    schedule = {0:{}}
    smart_order = sorted(list(counter.keys()),key=lambda x:counter[x]['effective'])
    # print('smart_order :',smart_order)
    schedule[0]['smart_order'] = smart_order
    schedule[0]['subcircuit_state'] = {}
    schedule[0]['upper_bin'] = None

    subcircuit_capacities = [counter[subcircuit_idx]['effective'] for subcircuit_idx in smart_order]
    # print('subcircuit_capacities:',subcircuit_capacities)
    if sum(subcircuit_capacities)<=recursion_qubit:
        subcircuit_active_qubits = distribute_load(total_load=sum(subcircuit_capacities),capacities=subcircuit_capacities)
    else:
        subcircuit_active_qubits = distribute_load(total_load=recursion_qubit,capacities=subcircuit_capacities)
    # print('subcircuit_active_qubits:',subcircuit_active_qubits)
    for subcircuit_idx, subcircuit_active_qubit in zip(smart_order,subcircuit_active_qubits):
        num_zoomed = 0
        num_active = subcircuit_active_qubit
        num_merged = counter[subcircuit_idx]['effective'] - num_zoomed - num_active
        schedule[0]['subcircuit_state'][subcircuit_idx] = ['active']*num_active + ['merged']*num_merged
    if verbose:
        [print(x,schedule[0][x],flush=True) for x in schedule[0]]
    return schedule

def next_dynamic_definition_schedule(recursion_layer,schedule,state_idx,recursion_qubit,verbose):
    if verbose:
        print('Initializing DD recursion, recursion_qubit=%d.'%recursion_qubit,flush=True)
    num_active = 0
    for subcircuit_idx in schedule['subcircuit_state']:
        num_active += schedule['subcircuit_state'][subcircuit_idx].count('active')
    bin_state_idx = bin(state_idx)[2:].zfill(num_active)
    # print('bin_state_idx = %s'%(bin_state_idx))
    bin_state_idx_ptr = 0
    for subcircuit_idx in schedule['smart_order']:
        for qubit_ctr, qubit_state in enumerate(schedule['subcircuit_state'][subcircuit_idx]):
            if qubit_state=='active':
                schedule['subcircuit_state'][subcircuit_idx][qubit_ctr] = bin_state_idx[bin_state_idx_ptr]
                bin_state_idx_ptr += 1
    schedule['smart_order'] = sorted(schedule['smart_order'],key=lambda x:schedule['subcircuit_state'][x].count('merged'))
    subcircuit_capacities = [schedule['subcircuit_state'][subcircuit_idx].count('merged') for subcircuit_idx in schedule['smart_order']]
    # print('subcircuit_capacities:',subcircuit_capacities)
    if sum(subcircuit_capacities)<=recursion_qubit:
        subcircuit_active_qubits = distribute_load(total_load=sum(subcircuit_capacities),capacities=subcircuit_capacities)
    else:
        subcircuit_active_qubits = distribute_load(total_load=recursion_qubit,capacities=subcircuit_capacities)
    # print('subcircuit_active_qubits:',subcircuit_active_qubits)
    for subcircuit_idx, subcircuit_active_qubit in zip(schedule['smart_order'],subcircuit_active_qubits):
        for qubit_ctr, qubit_state in enumerate(schedule['subcircuit_state'][subcircuit_idx]):
            if qubit_state=='merged' and subcircuit_active_qubit>0:
                schedule['subcircuit_state'][subcircuit_idx][qubit_ctr] = 'active'
                subcircuit_active_qubit -= 1
    schedule['upper_bin'] = (recursion_layer,state_idx)
    if verbose:
        [print(x,schedule[x],flush=True) for x in schedule]
    return schedule

def find_max_recursion_layer(curr_recursion_layer,dest_folder,meta_data):
    max_subgroup_prob = 0
    max_recursion_layer = -1
    for recursion_layer in range(curr_recursion_layer):
        dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,recursion_layer)
        build_output = read_dict(filename='%s/build_output.pckl'%(dynamic_definition_folder))
        zoomed_ctr = build_output['zoomed_ctr']
        max_states = build_output['max_states']
        reconstructed_prob = build_output['reconstructed_prob']
        schedule = meta_data['dynamic_definition_schedule'][recursion_layer]
        num_merged = 0
        for subcircuit_idx in schedule['subcircuit_state']:
            num_merged += schedule['subcircuit_state'][subcircuit_idx].count('merged')
        if num_merged==0 or zoomed_ctr==len(max_states):
            continue
        # print('Examine recursion_layer %d, zoomed_ctr = %d, max_state = %d, p = %e'%(
        #     recursion_layer,zoomed_ctr,max_states[zoomed_ctr],reconstructed_prob[max_states[zoomed_ctr]]))
        # print(schedule)
        if reconstructed_prob[max_states[zoomed_ctr]]>max_subgroup_prob and reconstructed_prob[max_states[zoomed_ctr]]>1e-16:
            max_subgroup_prob = reconstructed_prob[max_states[zoomed_ctr]]
            max_recursion_layer = recursion_layer
    return max_recursion_layer

def generate_meta_data(recursion_layer,counter,recursion_qubit,dest_folder,verbose):
    if recursion_layer==0:
        dynamic_definition_schedule = initialize_dynamic_definition_schedule(counter=counter,recursion_qubit=recursion_qubit,verbose=verbose)
        return {'counter':counter,'dynamic_definition_schedule':dynamic_definition_schedule}
    else:
        meta_data = read_dict(filename='%s/meta_data.pckl'%(dest_folder))
        max_recursion_layer = find_max_recursion_layer(curr_recursion_layer=recursion_layer,dest_folder=dest_folder,meta_data=meta_data)
        if max_recursion_layer==-1:
            print('-'*50,'DD recursions DONE','-'*50,flush=True)
            return None
        dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,max_recursion_layer)
        build_output = read_dict(filename='%s/build_output.pckl'%(dynamic_definition_folder))
        zoomed_ctr = build_output['zoomed_ctr']
        max_states = build_output['max_states']
        reconstructed_prob = build_output['reconstructed_prob']
        schedule = meta_data['dynamic_definition_schedule'][max_recursion_layer]
        print('Zoom in for results of recursion_layer %d'%max_recursion_layer,schedule,flush=True)
        print('state_idx = %d, p = %e'%(max_states[zoomed_ctr],reconstructed_prob[max_states[zoomed_ctr]]),flush=True)
        next_schedule = next_dynamic_definition_schedule(recursion_layer=max_recursion_layer,
        schedule=copy.deepcopy(schedule),state_idx=max_states[zoomed_ctr],recursion_qubit=recursion_qubit,verbose=verbose)
        build_output['zoomed_ctr'] += 1
        pickle.dump(build_output, open('%s/build_output.pckl'%(dynamic_definition_folder),'wb'))

        meta_data['dynamic_definition_schedule'][recursion_layer] = next_schedule
        return meta_data

def write_files(recursion_layer,num_threads,counter,vertical_collapse_folder,dynamic_definition_folder,dynamic_definition_schedule,summation_terms,num_cuts,num_subcircuits,verbose):
    for rank in range(num_threads):
        all_rank_kron_files = []
        for subcircuit_idx in counter:
            kron_files = glob.glob('%s/kron_%d_*.txt'%(vertical_collapse_folder,subcircuit_idx))
            rank_kron_files = find_process_jobs(jobs=kron_files,rank=rank,num_workers=num_threads)
            all_rank_kron_files += rank_kron_files
        merge_file = open('%s/merge_%d.txt'%(dynamic_definition_folder,rank),'w')
        merge_file.write('num_files_to_merge=%d\n'%(len(all_rank_kron_files)))
        for rank_kron_file in all_rank_kron_files:
            subcircuit_idx, subcircuit_kron_index = [int(x) for x in rank_kron_file.split('kron_')[-1].split('.txt')[0].split('_')]
            merge_file.write('subcircuit_idx=%d subcircuit_kron_index=%d num_effective=%d num_active=%d\n'%(
                subcircuit_idx,subcircuit_kron_index,counter[subcircuit_idx]['effective'],
                dynamic_definition_schedule[recursion_layer]['subcircuit_state'][subcircuit_idx].count('active')))
            for qubit_state in dynamic_definition_schedule[recursion_layer]['subcircuit_state'][subcircuit_idx]:
                if qubit_state=='merged':
                    merge_file.write('%d '%(-2))
                elif qubit_state=='active':
                    merge_file.write('%d '%(-1))
                elif qubit_state=='0' or qubit_state=='1':
                    merge_file.write('%s '%(qubit_state))
                else:
                    raise Exception('Illegal qubit_state %s'%qubit_state)
            merge_file.write('\n')
        merge_file.close()

        rank_summation_terms = find_process_jobs(jobs=summation_terms,rank=rank,num_workers=num_threads)
        num_summation_terms = len(rank_summation_terms)
        if verbose:
            print('Rank %d has %d/%d summation terms'%(rank,num_summation_terms,len(summation_terms)),flush=True)
        
        summation_term_file = open('%s/build_%d.txt'%(dynamic_definition_folder,rank),'w')
        total_active = 0
        for subcircuit_idx in dynamic_definition_schedule[recursion_layer]['subcircuit_state']:
            total_active += dynamic_definition_schedule[recursion_layer]['subcircuit_state'][subcircuit_idx].count('active')
        summation_term_file.write('total_active_qubit=%d num_subcircuits=%d num_summation_terms=%d num_cuts=%d\n'%(
            total_active,num_subcircuits,num_summation_terms,num_cuts))
        smart_order = dynamic_definition_schedule[recursion_layer]['smart_order']
        for summation_term in rank_summation_terms:
            for subcircuit_idx in smart_order:
                subcircuit_kron_index = summation_term[subcircuit_idx]
                summation_term_file.write('%d,%d '%(subcircuit_idx,subcircuit_kron_index))
            summation_term_file.write('\n')
        summation_term_file.close()

def distribute(circuit_name,max_subcircuit_qubit,eval_mode,early_termination,num_threads,qubit_limit,recursion_layer,recursion_qubit,verbose):
    source_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
    early_termination=None,eval_mode=None,num_threads=None,qubit_limit=None,field='cutter')
    eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
    early_termination=None,eval_mode=eval_mode,num_threads=None,qubit_limit=None,field='evaluator')
    vertical_collapse_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
    early_termination=early_termination,num_threads=None,eval_mode=eval_mode,qubit_limit=None,field='vertical_collapse')
    dest_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
    early_termination=early_termination,num_threads=num_threads,eval_mode=eval_mode,qubit_limit=qubit_limit,field='build')

    case_dict = read_dict(filename='%s/subcircuits.pckl'%source_folder)
    all_indexed_combinations = read_dict(filename='%s/all_indexed_combinations.pckl'%(eval_folder))
    if len(case_dict)==0:
        return
    
    full_circuit = case_dict['circuit']
    subcircuits = case_dict['subcircuits']
    complete_path_map = case_dict['complete_path_map']
    counter = case_dict['counter']
    num_subcircuits = len(subcircuits)

    meta_data = generate_meta_data(recursion_layer=recursion_layer,counter=counter,recursion_qubit=recursion_qubit,dest_folder=dest_folder,verbose=verbose)
    pickle.dump(meta_data, open('%s/meta_data.pckl'%(dest_folder),'wb'))

    O_rho_pairs, combinations = get_combinations(complete_path_map=complete_path_map)
    num_cuts = len(O_rho_pairs)

    _, summation_terms = build(full_circuit=full_circuit, combinations=combinations,
    O_rho_pairs=O_rho_pairs, subcircuits=subcircuits, all_indexed_combinations=all_indexed_combinations)

    dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,recursion_layer)
    if os.path.exists(dynamic_definition_folder):
        subprocess.run(['rm','-r',dynamic_definition_folder])
    os.makedirs(dynamic_definition_folder)

    write_files(recursion_layer=recursion_layer,num_threads=num_threads,counter=counter,
    vertical_collapse_folder=vertical_collapse_folder,dynamic_definition_folder=dynamic_definition_folder,
    dynamic_definition_schedule=meta_data['dynamic_definition_schedule'],summation_terms=summation_terms,num_cuts=num_cuts,num_subcircuits=num_subcircuits,
    verbose=verbose)