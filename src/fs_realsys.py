"""
    author: Suhas Vittal
    date:   11 April 2022
"""

from qiskit import Aer, IBMQ
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, InstructionDurations, CouplingMap
from qiskit.transpiler.passes import *

from qiskit.circuit.library import XGate

from fs_noise import *
from fs_util import G_QISKIT_GATE_SET

import pickle

import os
from collections import defaultdict

IBMQ.enable_account('f0f61055f98741e1e793cc5e0dddbb89567e59362c7ec34687938a3fe50cb765d6749943e8e41ed14fe9798c1663adf7bc0cfa6389f272c54765833936e7c713')
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc440')
qasmsim = Aer.get_backend('qasm_simulator')

def backend_to_arch_file(backend_name):
    print(provider.backends())
    ibmq_backend = provider.get_backend(backend_name)
    coupling_map = CouplingMap(ibmq_backend.configuration().coupling_map)
    writer = open('../arch/%s.arch' % backend_name, 'w')
    writer.write('%d\n' % coupling_map.size())
    for edge in coupling_map.get_edges():
        i,j = edge
        writer.write('%d %d\n' % (i,j))
    writer.close()

def run_circuits_on_device(backend_name, circuits=None):
    ibmq_backend = provider.get_backend(backend_name) 
    ins_dur = InstructionDurations.from_backend(ibmq_backend) 
    folder = '../benchmarks/fidelity_tests/%s' % backend_name

    if circuits is None:
        benchmark_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    else:
        benchmark_folders = circuits
    circ_array = []
    # Maintain dictionary to identify which results are what.
    circuit_to_index = defaultdict(dict)
    all_data = {}
    for subfolder in benchmark_folders:
        # Get base circuit
        base_circ = QuantumCircuit.from_qasm_file('%s/%s/base_mapping.qasm' % (folder, subfolder))
        base_counts = qasmsim.run(base_circ, shots=40000).result().get_counts()

        data = {'base counts': base_counts}

        print(subfolder)
        for cat in ['sabre','foresight_alap','noisy_foresight_alap',\
                'foresight_asap','noisy_foresight_asap']:
            circ = QuantumCircuit.from_qasm_file('%s/%s/%s_circ.qasm'\
                                                % (folder, subfolder, cat))
            circ = transpile(
                circ,
                backend=ibmq_backend,
                layout_method='trivial',
                routing_method='none',
                optimization_level=3
            )
            # Perform dynamical decoupling
            pm_dd = PassManager([ALAPScheduleAnalysis(ins_dur),\
                        PadDynamicalDecoupling(ins_dur, [XGate(), XGate()])])
            print('\t%s: cnots=%d, depth=%d' % (cat, circ.count_ops()['cx'], circ.depth()))
#            circ = pm_dd.run(circ)
            curr_index = len(circ_array)
            circuit_to_index[subfolder][cat] = curr_index
            for _ in range(5):
                circ_array.append(circ)
        all_data[subfolder] = data
    # Schedule all the circuits in the array.
    job_shots = ibmq_backend.configuration().max_shots
    job = ibmq_backend.run(circ_array, shots=job_shots)
    res = job.result()
    for subfolder in benchmark_folders:
        print(subfolder)
        for cat in ['sabre','foresight_alap','noisy_foresight_alap',\
                'foresight_asap','noisy_foresight_asap']:
            job_index = circuit_to_index[subfolder][cat]
            counts = defaultdict(int)
            for i in range(5):
                c = res.get_counts(job_index+i)
                for x in c:
                    counts[x] += c[x]
            base_counts = all_data[subfolder]['base counts']
            fidelity = get_evaluation_output(base_counts, counts, metric='fidelity')
            ist = get_evaluation_output(base_counts, counts, metric='ist')
            print('\t%s: fidelity=%f, ist=%f' % (cat, fidelity, ist))

            all_data[subfolder][cat] = {
                'counts': counts,
                'fidelity': fidelity,
                'ist': ist
            }
        data = all_data[subfolder]
        # Write to pickle.
        writer = open('%s/%s/counts.pkl' % (folder, subfolder), 'wb')
        pickle.dump(data, writer)
        writer.close()

if __name__ == '__main__':
    from sys import argv

    from qiskit.transpiler import CouplingMap

    backend_name = argv[1]
    run_circuits_on_device(backend_name)

