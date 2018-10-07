# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

PACKAGE_NAME = 'pysaliency'
VERSION = '0.2.0'
DESCRIPTION = 'A Python Framework for Saliency Modeling and Evaluation'
AUTHOR = 'Matthias Kümmerer'
EMAIL = 'matthias.kuemmerer@bethgelab.org'
URL = "https://github.com/matthiask/pysaliency"

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

extensions = [
    Extension("pysaliency.roc", ['pysaliency/*.pyx'],
              include_dirs = [np.get_include()],
              extra_compile_args = ['-fopenmp', '-O3'],
              extra_link_args=["-fopenmp"]),
]


setup(
    name = PACKAGE_NAME,
    version = VERSION,
    packages = find_packages(),
	author = AUTHOR,
	author_email = EMAIL,
	url = URL,
	license = 'MIT',
	install_requires=[
		'boltons',
		'imageio',
		'natsort',
		'numpy',
		'requests',
		'scipy',
		'setuptools',
	    'tqdm',
	],
    include_package_data = True,
    package_data={'pysaliency': ['scripts/*.m',
                                 'scripts/models/*.m',
                                 'scripts/models/*/*.m',
                                 'scripts/models/*/*/*',
                                 'scripts/models/BMS/patches/*',
                                 'scripts/models/GBVS/patches/*',
                                 'scripts/models/Judd/patches/*',
                                 ]},
    ext_modules = cythonize(extensions),
)
