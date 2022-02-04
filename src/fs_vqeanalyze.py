"""
    author: Suhas Vittal
    date:   3 February 2022
"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import * 

from fs_statsabre import StatSABRE
from fs_foresight import ForeSight
from fs_exec import _pad_circuit_to_fit
from fs_util import *

import matplotlib.pyplot as plt
import numpy as np

import pickle as pkl

DEFAULT_OUTPUT_FOLDER = 'vqeanalysis'

def vqe_swapsegment_plotgen(coupling_map, foresight, sabre, runs=3, output_folder=DEFAULT_OUTPUT_FOLDER):
    benchmark_folder, benchmarks = G_VQE 

    mapping_pass = PassManager([
        SabreLayout(coupling_map, routing_pass=SabreSwap(coupling_map, heuristic='decay'),max_iterations=3),
        ApplyLayout() 
    ])

    basis_pass = Unroller(G_QISKIT_GATE_SET)
    foresight_pass = PassManager([
        basis_pass,
        foresight,
        basis_pass
    ])
    sabre_pass = PassManager([
        basis_pass,
        sabre,
        basis_pass
    ])

    routing_list = [
        ('ForeSight', foresight_pass, foresight),
        ('SABRE', sabre_pass, sabre)
    ]

    data = {}
    for qbfile in benchmarks:
        circ = QuantumCircuit.from_qasm_file('%s/%s' % (benchmark_folder, qbfile))
        print('[%s]' % qbfile)
        _pad_circuit_to_fit(circ, coupling_map)

        min_swap_segments = {} 
        cumulative_swap_segments = {}
        min_cnots = {}
        for _ in range(runs):
            mapped_circ = mapping_pass.run(circ)
            for (name,routing_pass,policy) in routing_list:
                final_circ = routing_pass.run(mapped_circ)
                cnots = final_circ.count_ops()['cx'] - circ.count_ops()['cx']
                if name not in min_cnots or cnots < min_cnots[name]:
                    min_cnots[name] = cnots
                    min_swap_segments[name] = policy.swap_segments 
            # Compute cumulative swaps
        for name in min_swap_segments:
            cumulative_swaps = 0
            segment = []
            for s in min_swap_segments[name]:
                cumulative_swaps += s
                segment.append(cumulative_swaps)
            cumulative_swap_segments[name] = segment
            print('CNOTs for %s: %d' % (name,min_cnots[name]))
        data[qbfile] = min_swap_segments
        # Create plots as well
        for name in min_swap_segments:
            swap_segment = min_swap_segments[name]
            cswap_segment = cumulative_swap_segments[name]
            xaxis = np.arange(len(swap_segment))
            plt.plot(xaxis, swap_segment, 'o', label='Swaps by Segment, %s' % name)
        plt.xlabel('Swap Segment Number')
        plt.ylabel('Number of Swaps')
        plt.legend()
        plt.savefig('%s/vqe_swap_segments_%s.png' % (output_folder,qbfile))
        plt.clf()
        for name in min_swap_segments:
            cswap_segment = cumulative_swap_segments[name]
            xaxis = np.arange(len(cswap_segment))
            plt.plot(xaxis, cswap_segment, 'o', label='Cumulative Swaps, %s' % name) 
        plt.xlabel('Swap Segment Number')
        plt.ylabel('Number of Swaps')
        plt.legend()
        plt.savefig('%s/vqe_cumulative_swaps_%s.png' % (output_folder,qbfile))
        plt.clf()
    writer = open('%s/vqeanalysis.pkl' % output_folder,'wb') 
    pkl.dump(data,writer)
    writer.close()

if __name__ == '__main__':
    coupling_map = G_GOOGLE_WEBER
    foresight = ForeSight(coupling_map, slack=3, solution_cap=32)
    sabre = StatSABRE(coupling_map, heuristic='decay')
    vqe_swapsegment_plotgen(coupling_map, foresight, sabre, runs=3)
