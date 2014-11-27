from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#import sys

#if sys.version_info < (2, 7):
#    version_ext = '_26'
#else:
#    version_ext = '_27'

#roc_module = Extension(
#    "roc",
#    ["roc.pyx"],
#    #extra_compile_args=['-fopenmp', '-O3'],
#    #extra_link_args=['-fopenmp'],
#)

def enable_openmp(ext):
    ext.extra_compile_args.extend(['-fopenmp', '-O3'])
    ext.extra_link_args.extend(['-fopenmp'])
    return ext

def openmp_cythonize(*args, **kwargs):
    exts = cythonize(*args, **kwargs)
    for e in exts:
        enable_openmp(e)
    return exts

setup(
    name = 'pysaliency',
    ext_modules = openmp_cythonize("pysaliency/*.pyx"),
)
