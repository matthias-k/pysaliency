# -*- coding: utf-8 -*-
from os import path

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
import io

PACKAGE_NAME = 'pysaliency'
VERSION = '0.2.22'
DESCRIPTION = 'A Python Framework for Saliency Modeling and Evaluation'
AUTHOR = 'Matthias Kümmerer'
EMAIL = 'matthias.kuemmerer@bethgelab.org'
URL = "https://github.com/matthiask/pysaliency"

try:
    this_directory = path.abspath(path.dirname(__file__))
    with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except IOError:
    long_description = ''

extensions = [
    Extension("pysaliency.roc_cython", ['pysaliency/*.pyx'],
              include_dirs = [np.get_include()],
              extra_compile_args = ['-O3'],
              #extra_compile_args = ['-fopenmp', '-O3'],
              #extra_link_args=["-fopenmp"]
              ),
]


setup(
    name = PACKAGE_NAME,
    version = VERSION,
    description = 'python library to develop, evaluate and benchmark saliency models',
    long_description = long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        #"Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        #"Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    packages = find_packages(),
    author = AUTHOR,
    author_email = EMAIL,
    url = URL,
    license = 'MIT',
    install_requires=[
        'boltons',
        'deprecation',
        'imageio',
        'natsort',
        'numba',
        'numpy',
        'piexif',
        'requests',
        'schema',
        'scipy',
        'setuptools',
        'tqdm',
    ],
    include_package_data = True,
    package_data={'pysaliency': ['external_models/scripts/*.m',
                                 'external_models/scripts/*/*.m',
                                 'external_models/scripts/*/*/*',
                                 'external_models/scripts/BMS/patches/*',
                                 'external_models/scripts/GBVS/patches/*',
                                 'external_models/scripts/Judd/patches/*',
                                 'external_datasets/scripts/*.m'
                                 ]},
    ext_modules = cythonize(extensions),
)
