"""
    author: Suhas Vittal
    date:   11 April 2022
"""

from fs_util import read_arch_file
from fs_benchmark import benchmark_circuits
from fs_benchmark import _sabre_route, _foresight_route, _astar_route, _tket_route
from fs_foresight import *
from fs_noise import google_sycamore_noise_model

IBM_TOKYO = read_arch_file('../arch/ibm_tokyo.arch')
GOOGLE_SYCAMORE = read_arch_file('../arch/google_weber.arch')
RIGETTI_ASPEN9 = read_arch_file('../arch/rigetti_aspen9.arch')
GRID500 = read_arch_file('../arch/500grid.arch')

def batch1():
    # SABRE: Tokyo
    # ForeSight: Sycamore
    # A*: Rigetti
    # TKET: Rigetti
    foresight = ForeSight(
        GOOGLE_SYCAMORE,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'foresight',
        _foresight
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'astar',
        _astar_route
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'tket',
        _tket_route
    )

def batch2():
    # SABRE: Sycamore
    # ForeSight: Rigetti
    # A*: Tokyo
    # TKET: Tokyo
    foresight = ForeSight(
        RIGETTI_ASPEN9,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'foresight',
        _foresight
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'astar',
        _astar_route
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'tket',
        _tket_route
    )

def batch3():
    # SABRE: Rigetti
    # ForeSight: Tokyo
    # A*: Sycamore
    # TKET: Sycamore
    foresight = ForeSight(
        IBM_TOKYO,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'foresight',
        _foresight
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'astar',
        _astar_route
    )
    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'tket',
        _tket_route
    )

def batch4():  # General sensitivity analysis
    # ForeSight: (delta, S) = (0, 64), (1, 64), (2, 64), (3, 64), (4, 64)
    #                       = (2, 4), (2, 8), (2, 16), (2, 32), (2, 64)
    def genforesight(slack, solution_cap):
        foresight = ForeSight(
            GOOGLE_SYCAMORE,
            slack=slack,
            solution_cap=solution_cap,
            flags=FLAG_ALAP
        )
        return (lambda x,y: _foresight_route(x,y,foresight))
    foresight_table = {}
    for slack in [0,1,2,3,4]:
        for solution_cap in [4,8,16,32,64]:
            _foresight = genforesight(slack,solution_cap)
            foresight_table[(slack,solution_cap)] = _foresight
    for slack in [0,1,2,3,4]:
        for solution_cap in [4,8,16,32,64]:
            _foresight = foresight_table[(slack,solution_cap)]
            benchmark_circuits(
                '../benchmarks/gensens',
                '../arch/google_weber.arch',
                'fs_%d_%d' % (slack, solution_cap),
                _foresight
            )

def batch5():   # BV sensitivity analysis 1
    def genforesight(slack, solution_cap):
        foresight = ForeSight(
            GOOGLE_SYCAMORE,
            slack=slack,
            solution_cap=solution_cap,
            flags=FLAG_ALAP
        )
        return (lambda x,y: _foresight_route(x,y,foresight))
    foresight_table = {}
    for slack in [0,1,2,3,4]:
        for solution_cap in [4,8,16,32,64]:
            _foresight = genforesight(slack,solution_cap)
            foresight_table[(slack,solution_cap)] = _foresight
    for slack in [0,1,2,3,4]:
        for solution_cap in [4,8,16,32,64]:
            _foresight = foresight_table[(slack,solution_cap)]
            benchmark_circuits(
                '../benchmarks/bvsens1',
                '../arch/google_weber.arch',
                'fs_%d_%d' % (slack, solution_cap),
                _foresight
            )

def batch6():   # BV sensitivity analysis 2
    foresight = ForeSight(
        GRID500,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight = lambda x,y: _foresight_route(x,y,foresight)
    benchmark_circuits(
        '../benchmarks/bvsens2',
        '../arch/500grid.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/bvsens2',
        '../arch/500grid.arch',
        'foresight',
        _foresight
    )

def batch7():   # Noise simulation routing
    _, cx_error_rates, sq_error_rates, ro_error_rates =\
        google_sycamore_noise_model(GOOGLE_SYCAMORE, '../arch/noisy/google_weber.noise')
    foresight_noise_unaware = ForeSight(
        GOOGLE_SYCAMORE,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    foresight_noise_aware = ForeSight(
        GOOGLE_SYCAMORE,
        slack=0.01,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        flags=FLAG_ALAP|FLAG_NOISE_AWARE
    )
    _fs1 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware)
    _fs2 = lambda x,y: _foresight_route(x,y,foresight_noise_aware)
    benchmark_circuits(
        '../benchmarks/noise',
        '../arch/google_weber.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/noise',
        '../arch/google_weber.arch',
        'foresight',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/noise',
        '../arch/google_weber.arch',
        'noisy_foresight',
        _fs2,
        runs=1
    )

def batch8():   # SK Models routing
    archs = ['../arch/google_weber.arch']
    arch_names = ['google_sycamore']
    for (i,arch_file) in enumerate(archs):
        print(arch_names[i])
        backend = read_arch_file(arch_file)
        benchmark_circuits(
            '../benchmarks/skbench/%s' % arch_names[i],
            arch_file,
            'sabre',
            _sabre_route
        )
        foresight_asap = ForeSight(backend, slack=2, solution_cap=64,
                flags=FLAG_ASAP|FLAG_OPT_FOR_O3)
        _fs = lambda x,y: _foresight_route(x,y,foresight_asap)
        benchmark_circuits(
            '../benchmarks/skbench/%s' % arch_names[i],
            arch_file,
            'foresight_asap',
            _fs,
            runs=1
        )

def batch9():
    archs = ['../arch/google_weber.arch']
    arch_names = ['google_sycamore']
    for (i,arch_file) in enumerate(archs):
        print(arch_names[i])
        backend = read_arch_file(arch_file)
        benchmark_circuits(
            '../benchmarks/irrbench/%s' % arch_names[i],
            arch_file,
            'sabre',
            _sabre_route
        )
        foresight_asap = ForeSight(backend, slack=2, solution_cap=64,
                flags=FLAG_ASAP|FLAG_OPT_FOR_O3)
        _fs = lambda x,y: _foresight_route(x,y,foresight_asap)
        benchmark_circuits(
            '../benchmarks/irrbench/%s' % arch_names[i],
            arch_file,
            'foresight_asap',
            _fs,
            runs=1
        )

if __name__ == '__main__':
    from sys import argv
    batch_no = int(argv[1])
    exec('batch%d()' % batch_no)

