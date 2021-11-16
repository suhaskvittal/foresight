"""
    author: Suhas Vittal
    date: 15 November 2021
"""
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from qiskit.quantum_info import Operator
from qiskit.extensions import UnitaryGate

import numpy as np

from fs_util import * 

def google_weber_noise_model():
    weber = G_GOOGLE_WEBER 
    # Noise parameters from Weber (some estimated)
    G1Q_TIME = 25  # ns
    CX_TIME = 32 
    MEAS_TIME = 4000
    T1 = 15000
    T2 = T1*0.5
    # Read noise file.
    noise_file = 'arch/noisy/google_weber.noise'
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
        q, e_1q, pm1g0, pm0g1 = int(split_line[0]), float(split_line[1]), float(split_line[2]), float(split_line[3])
        for g in g_1q_err:
            g_1q_err[g][q] = e_1q
            g_1q_time[g][q] = G1Q_TIME
        prob_m1_g0[q] = pm1g0
        prob_m0_g1[q] = pm0g1
        meas_time[q] = MEAS_TIME
        coh_t1[q] = T1
        coh_t2[q] = T2
    line = reader.readline()  # now read until the end of the file
    while line != '':
        split_line = line.split(' ')
        q1, q2, e_2q = int(split_line[0]), int(split_line[1]), float(split_line[2])
        g_2q_err['cx'][(q1,q2)] = e_2q 
        g_2q_err['cx'][(q2,q1)] = e_2q
        g_2q_time['cx'][(q1,q2)] = CX_TIME
        g_2q_time['cx'][(q2,q1)] = CX_TIME
        line = reader.readline()
    noise_model = _build_noise_model(
        num_qubits,
        weber,
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
        use_readout=False
    )
    # Build edge weights for ForeSight
    edge_weights = {}
    for (i,j) in weber.get_edges():
        if (i,j) in g_2q_err['cx']:
            e = g_2q_err['cx'][(i,j)]
        else:
            e = g_2q_err['cx'][(j,i)]
        # Log weighting maintains product property
        edge_weights[(i,j)] = -np.log(1-e)
    # Build vertex weights and readout weights for ForeSight
    vertex_weights = {}
    readout_weights = {}
    for i in range(num_qubits):
        mean_1qe = np.mean([
            g_1q_err[g][i] for g in g_1q_err
        ])
        mean_mse = 0.5*(prob_m1_g0[i] + prob_m0_g1[i])
        vertex_weights[i] = mean_1qe 
        readout_weights[i] = mean_mse
    return noise_model, edge_weights, vertex_weights, None#readout_weights

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
    use_readout=True
):
    noise_model = NoiseModel() 
    if use_1qerror:
        # Add single qubit error
        for i in range(num_qubits):
            for gate in g_1q_err:
                # Gate error
                e = depolarizing_error(g_1q_err[gate][i], 1)
                noise_model.add_quantum_error(e, [gate], [i], warnings=warnings) 
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
                # Coherence error
                e = thermal_relaxation_error(coh_t1[i], coh_t2[i], g_2q_time[gate][(i,j)]).expand(
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
            # Coherence error
            e = thermal_relaxation_error(coh_t1[i], coh_t2[i], meas_time[i])
            noise_model.add_quantum_error(e, ['measure'], [i], warnings=warnings)
    return noise_model

def _truncate(n, decimals=0):
    f = 10**decimals
    return np.round(n*f) / f

