# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

PACKAGE_NAME = 'pysaliency'
VERSION = '0.1.3'
DESCRIPTION = 'A Python Framework for Saliency Modeling and Evaluation'
AUTHOR = 'Matthias Kümmerer'
EMAIL = 'matthias.kuemmerer@bethgelab.org'
URL = "https://github.com/matthiask/pysaliency"

extensions = [
    Extension("pysaliency.roc", ['pysaliency/*.pyx'],
              include_dirs = [np.get_include()],
              extra_compile_args = ['-fopenmp', '-O3'],
              extra_link_args=["-fopenmp"]),
]


setup(
    name = PACKAGE_NAME,
    version = VERSION,
    packages = [PACKAGE_NAME],
	author = AUTHOR,
	author_email = EMAIL,
	url = URL,
	license = 'MIT',
	install_requires=[
		'boltons',
		'natsort',
		'numpy',
		'requests',
		'scipy',
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
