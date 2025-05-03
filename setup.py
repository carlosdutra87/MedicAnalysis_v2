# cython_kernels/setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="cython_kernels.kernels",
    sources=["cython_kernels/kernels.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="cython_kernels",
    ext_modules=cythonize([ext], language_level=3),
)
