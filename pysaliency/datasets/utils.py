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


def hdf5_wrapper(mode=None):
    def decorator(f):
        @wraps(f)
        def wrapped(self, target, *args, **kwargs):
            if isinstance(target, (str, pathlib.Path)):
                import h5py
                with h5py.File(target, mode) as hdf5_file:
                    return f(self, hdf5_file, *args, **kwargs)
            else:
                return f(self, target, *args, **kwargs)

        return wrapped
    return decorator


def decode_string(data):
    if not isinstance(data, str):
        return data.decode('utf8')

    return data

def create_hdf5_dataset(target, name, data):
    import h5py

    if isinstance(np.array(data).flatten()[0], str):
        data = np.array(data)
        original_shape = data.shape
        encoded_items = [decode_string(item).encode('utf8') for item in data.flatten()]
        encoded_data = np.array(encoded_items).reshape(original_shape)

        target.create_dataset(
            name,
            data=encoded_data,
            dtype=h5py.special_dtype(vlen=str)
        )
    else:
        target.create_dataset(name, data=data)