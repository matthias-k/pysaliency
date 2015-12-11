# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

PACKAGE_NAME = 'pysaliency'
VERSION = '0.1.0'
DESCRIPTION = 'A Python Framework for Saliency Modeling and Evaluation'
AUTHOR = 'Matthias Kümmerer'
EMAIL = 'matthias.kuemmerer@bethgelab.org'
URL = "https://github.com/matthiask/pysaliency"

extensions = [
    Extension("roc", ['pysaliency/*.pyx'],
              include_dirs = [np.get_include()],
              extra_compile_args = ['-fopenmp', '-O3'],
              extra_link_args=["-fopenmp"]),
]


setup(
    name = PACKAGE_NAME,
    version = VERSION,
    packages = [PACKAGE_NAME],
    include_package_data = True,
    package_data={'pysaliency': ['scripts/*.m',
                                 'scripts/models/*.m',
                                 'scripts/models/*/*.m'
                                 'scripts/models/BMS/patches/*'
                                 'scripts/models/GBVS/patches/*'
                                 'scripts/models/Judd/patches/*'
                                 ]},
    ext_modules = cythonize(extensions),
)
