from __future__ import absolute_import, print_function, division

import zipfile
import os
import glob

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    download_and_check,
    atomic_directory_setup,
)

from .utils import create_stimuli, _load

from .toronto import get_toronto, get_toronto_with_subjects
from .mit import get_mit1003, get_mit1003_with_initial_fixation, get_mit1003_onesize, get_mit300
from .cat2000 import get_cat2000_test, get_cat2000_train
from .isun import get_iSUN, get_iSUN_training, get_iSUN_validation, get_iSUN_testing
from .salicon import get_SALICON, get_SALICON_train, get_SALICON_val, get_SALICON_test
from .koehler import get_koehler
from .figrim import get_FIGRIM
from .osie import get_OSIE
from .nusef import get_NUSEF_public
from .pascal_s import get_PASCAL_S
from .dut_omrom import get_DUT_OMRON
from .coco_search18 import get_COCO_Search18, get_COCO_Search18_train, get_COCO_Search18_validation
from .coco_freeview import get_COCO_Freeview, get_COCO_Freeview_train, get_COCO_Freeview_validation, get_COCO_Freeview_test
