import json
import os
import pathlib
import warnings
from collections.abc import Sequence
from functools import wraps
from hashlib import sha1
from typing import Dict, List, Optional, Union
from weakref import WeakValueDictionary

import numpy as np
from boltons.cacheutils import cached

from ..utils.variable_length_array import VariableLengthArray, concatenate_variable_length_arrays

try:
    from imageio.v3 import imread
except ImportError:
    from imageio import imread
from PIL import Image
from tqdm import tqdm

from ..utils import LazyList, remove_trailing_nans