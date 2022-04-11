"""
    author: Suhas Vittal
    date:   7 April 2022
"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.compiler import transpile
from qiskit.transpiler.passes import *
from qiskit import Aer

from fs_foresight import *
from fs_util import read_arch_file, G_QISKIT_GATE_SET

from sys import argv

qasmsim = Aer.get_backend('qasm_simulator')

if __name__ == '__main__':
    circ_file = argv[1]
    arch_file = argv[2]
    slack = float(argv[3])
    solution_cap = int(argv[4])

    coupling_map = read_arch_file(arch_file)

    compiler = ForeSight(
        coupling_map=coupling_map,
        slack=slack,
        solution_cap=solution_cap,
        flags=FLAG_DEBUG | FLAG_ALAP# | FLAG_OPT_FOR_O3
    )
    foresight = PassManager([
        TrivialLayout(coupling_map),
        ApplyLayout(),
        compiler
    ])

    sabre = PassManager([
        TrivialLayout(coupling_map),
        ApplyLayout(),
        SabreSwap(coupling_map, heuristic='decay')
    ])

    circ = QuantumCircuit.from_qasm_file(circ_file)
    if 'measure' not in circ.count_ops():
        circ.measure_active()
    fs_circ = foresight.run(circ)
    sabre_circ = sabre.run(circ)
    fs_circ = sabre.run(fs_circ)

    # Check correctness of foresight circuit
    ideal_counts = qasmsim.run(circ).result().get_counts()
    fs_counts = qasmsim.run(fs_circ).result().get_counts()
    print(ideal_counts)
    print(fs_counts)

    print(fs_circ.count_ops())
    print(sabre_circ.count_ops())
    print(fs_circ.depth())
    print(sabre_circ.depth())

    writer = open('foresight_circ.qasm', 'w')
    writer.write(fs_circ.qasm())
    writer.close()
    writer = open('sabre_circ.qasm', 'w')
    writer.write(sabre_circ.qasm())
    writer.close()

    print('after O0')
    fs_circ = transpile(fs_circ,
                coupling_map=coupling_map,
                basis_gates=G_QISKIT_GATE_SET,
                layout_method='trivial',
                routing_method='none',
                optimization_level=0)
    sabre_circ = transpile(sabre_circ,
                coupling_map=coupling_map,
                basis_gates=G_QISKIT_GATE_SET,
                layout_method='trivial',
                routing_method='none',
                optimization_level=0)
    print(fs_circ.count_ops())
    print(sabre_circ.count_ops())
    print(fs_circ.depth())
    print(sabre_circ.depth())
    print('after O3')
    fs_circ = transpile(fs_circ,
                coupling_map=coupling_map,
                basis_gates=G_QISKIT_GATE_SET,
                layout_method='trivial',
                routing_method='none',
                optimization_level=3)
    sabre_circ = transpile(sabre_circ,
                coupling_map=coupling_map,
                basis_gates=G_QISKIT_GATE_SET,
                layout_method='trivial',
                routing_method='none',
                optimization_level=3)
    print(fs_circ.count_ops())
    print(sabre_circ.count_ops())
    print(fs_circ.depth())
    print(sabre_circ.depth())
