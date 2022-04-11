"""
    author: Suhas Vittal
    date:   11 April 2022
"""

from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile

from fs_noise import *
from fs_util import G_QISKIT_GATE_SET

import pickle

qasmsim = Aer.get_backend('qasm_simulator')

SHOTS=400000

def simulate(folder): 
    # NOTE: simulates a single qasm folder (i.e. adder_n10.qasm)
    backend = read_arch_file('../arch/google_weber.arch')
    noise_model, _, _, _ = google_sycamore_noise_model(backend,'../arch/noisy/google_weber.noise')

    base_circ = QuantumCircuit.from_qasm_file('%s/base_mapping.qasm' % folder)
    base_counts = qasmsim.run(base_circ, shots=SHOTS).result().get_counts()

    data = {
        'base counts': base_counts
    }
    for cat in ['sabre','foresight','noisy_foresight']:
        circ = QuantumCircuit.from_qasm_file('%s/%s_circ.qasm' % (folder, cat))
        circ = transpile(
            circ,
            basis_gates=G_QISKIT_GATE_SET,
            coupling_map=backend,
            layout_method='trivial',
            routing_method='none',
            optimization_level=3
        )
        counts = qasmsim.run(circ, shots=SHOTS,
                basis_gates=G_QISKIT_GATE_SET, noise_model=noise_model).result().get_counts()
        fidelity = get_evaluation_output(base_counts, counts, metric='fidelity')
        ist = get_evaluation_output(base_counts, counts, metric='ist')
        data[cat] = {
            'counts': counts,
            'fidelity': fidelity,
            'ist': ist
        }
    writer = open('%s/counts.pkl' % folder, 'w')
    pickle.dump(data, writer)
    writer.close()

if __name__ == '__main__':
    from sys import argv
    
    folder = argv[1]
    simulate(folder)
