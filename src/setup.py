from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'foresight',
        ['fs_foresight.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='foresight-omp',
    ext_modules=cythonize(ext_modules),
)
