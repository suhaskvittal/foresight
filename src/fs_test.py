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
from fs_benchmark import qiskitopt3_layout_pass
             

from sys import argv
import time

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
        flags=FLAG_DEBUG | FLAG_ASAP# | FLAG_OPT_FOR_O3
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
    if 'cx' not in circ.count_ops():
        base_cnots = 0
    else:
        base_cnots = circ.count_ops()['cx']
    if 'measure' not in circ.count_ops():
        circ.measure_active()
#    layout_pass = qiskitopt3_layout_pass(coupling_map,
#        routing_pass=ForeSight(coupling_map=coupling_map,slack=2,solution_cap=1,flags=FLAG_ASAP))
#    circ1 = layout_pass.run(circ)
    layout_pass = qiskitopt3_layout_pass(coupling_map, do_unroll=True)
    circ2 = layout_pass.run(circ)

    start = time.time()
    fs_circ = foresight.run(circ2)
    end = time.time()
    sabre_circ = sabre.run(circ2)

    print('foresight time taken:', (end-start)*1000)

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
    foresight_cnots = fs_circ.count_ops()['cx'] 
    sabre_cnots = sabre_circ.count_ops()['cx']

    print('foresight cnots', (foresight_cnots-base_cnots))
    print('sabre cnots', (sabre_cnots-base_cnots))
    print('foresight depth', fs_circ.depth())
    print('sabre depth', sabre_circ.depth())
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
    foresight_cnots = fs_circ.count_ops()['cx'] 
    sabre_cnots = sabre_circ.count_ops()['cx']

    print('foresight cnots', (foresight_cnots-base_cnots))
    print('sabre cnots', (sabre_cnots-base_cnots))
    print('foresight depth', fs_circ.depth())
    print('sabre depth', sabre_circ.depth())

