"""
    author: Suhas Vittal
    date: 22 November 2021
"""

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import *

from fs_util import *
from fs_noise import google_weber_noise_model
from fs_exec import _pad_circuit_to_fit
from fs_exec import *

from fs_foresight import ForeSight

import pickle as pkl
from sys import argv

def generate_qobjs():
    coupling_map = G_GOOGLE_WEBER
    _, edge_weights, _, _ = google_weber_noise_model(1.0)

    initial_pass = PassManager([
        Unroller(G_QISKIT_GATE_SET),
        SabreLayout(coupling_map, routing_pass=SabreSwap(coupling_map,heuristic='decay')),
        ApplyLayout()
    ])
    sabre_pass = PassManager([
        SabreSwap(coupling_map,heuristic='decay'),
        Unroller(G_QISKIT_GATE_SET)
    ])
    foresight_pass = PassManager([
        ForeSight(
            coupling_map,
            slack=G_FORESIGHT_SLACK,
            solution_cap=G_FORESIGHT_SOLN_CAP
        ),
        Unroller(G_QISKIT_GATE_SET)
    ])
    noisy_foresight_pass = PassManager([
        ForeSight(
            coupling_map,
            slack=0.01,
            solution_cap=G_FORESIGHT_SOLN_CAP,
            edge_weights=edge_weights
        ),
        Unroller(G_QISKIT_GATE_SET)
    ])

    data = {}
    folder, benchmarks = G_NOISY
    for qb_file in benchmarks:
        print(qb_file)
        circ = QuantumCircuit.from_qasm_file('%s/%s' % (folder, qb_file))    
        _pad_circuit_to_fit(circ, coupling_map)
        circ = initial_pass.run(circ)
        best_sabre_circ, best_foresight_circ, best_noisy_foresight_circ = None, None, None
        for r in range(3):
            print('\truns:', r)
            sabre_circ = sabre_pass.run(circ)
            foresight_circ = foresight_pass.run(circ)
            noisy_foresight_circ = noisy_foresight_pass.run(circ)
            if r == 0:
                best_sabre_circ = sabre_circ
                best_foresight_circ = foresight_circ
                best_noisy_foresight_circ = noisy_foresight_circ
            else:
                if best_sabre_circ.count_ops()['cx'] > sabre_circ.count_ops()['cx']:
                    best_sabre_circ = sabre_circ
                if best_foresight_circ.count_ops()['cx'] > foresight_circ.count_ops()['cx']:
                    best_foresight_circ = foresight_circ
                if best_noisy_foresight_circ.count_ops()['cx'] > noisy_foresight_circ.count_ops()['cx']:
                    best_noisy_foresight_circ = noisy_foresight_circ
        ideal_counts = exec_sim(circ)
        data[qb_file] = {
            'counts': ideal_counts,
            'sabre': best_sabre_circ,
            'foresight': best_foresight_circ,
            'noisy foresight': best_noisy_foresight_circ
        }
    with open('routed_qobjs/weber_circ.pkl', 'wb') as writer:
        pkl.dump(data, writer)

def noise_sweep(noise_factor):
    noise_model, edge_weights, _, _ = google_weber_noise_model(noise_factor)
    with open('routed_qobjs/weber_circ.pkl', 'rb') as reader:
        d = pkl.load(reader)
    sim_counts = {}
    for qbfile in G_QASMBENCH_MEDIUM:
        print(qbfile)
        sim_counts[qbfile] = {'ideal counts': d[qbfile]['counts']}
        for policy in ['sabre', 'foresight', 'noisy foresight']:
            circ = d[qbfile][policy]
            counts = exec_sim(circ, noise_model=noise_model)
            # compute eps
            eps = 1.0
            for (_, qargs, _) in circ.get_instructions('cx'):
                q0, q1 = qargs
                eps *= 1.0 - edge_weights[(q0.index, q1.index)]
            tvd = total_variation_distance(d[qbfile]['counts'], counts)
            sim_counts[qbfile][policy] = {
                'counts': counts,
                'eps': eps,
                'tvd': tvd
            }
    with open('routed_qobjs/weber_circ_sweep_%.8f.pkl' % noise_factor, 'wb') as writer:
        pkl.dump(sim_counts, writer)

if __name__ == '__main__':
    mode = argv[1]
    if mode == 'init':
        generate_qobjs() 
    else:
        noise_sweep(float(argv[2]))
