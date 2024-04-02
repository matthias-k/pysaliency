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


def get_merged_attribute_list(attributes):
    all_attributes = set(attributes[0])
    common_attributes = set(attributes[0])

    for _attributes in attributes[1:]:
        all_attributes = all_attributes.union(_attributes)
        common_attributes = common_attributes.intersection(_attributes)

    if common_attributes != all_attributes:
        lost_attributes = all_attributes.difference(common_attributes)
        warnings.warn(f"Discarding attributes which are not present everywhere: {lost_attributes}", stacklevel=4)

    return sorted(common_attributes)

def _load_attribute_dict_from_hdf5(attribute_group):
    json_attributes = attribute_group.attrs['__attributes__']
    if not isinstance(json_attributes, str):
        json_attributes = json_attributes.decode('utf8')
    __attributes__ = json.loads(json_attributes)

    attributes = {attribute: attribute_group[attribute][...] for attribute in __attributes__}
    return attributes


def get_merged_attribute_list(attributes):
    all_attributes = set(attributes[0])
    common_attributes = set(attributes[0])

    for _attributes in attributes[1:]:
        all_attributes = all_attributes.union(_attributes)
        common_attributes = common_attributes.intersection(_attributes)

    if common_attributes != all_attributes:
        lost_attributes = all_attributes.difference(common_attributes)
        warnings.warn(f"Discarding attributes which are not present everywhere: {lost_attributes}", stacklevel=4)

    return sorted(common_attributes)


def concatenate_attributes(attributes):
    attributes = list(attributes)

    if isinstance(attributes[0], VariableLengthArray):
        return concatenate_variable_length_arrays(attributes)

    attributes = [np.array(a) for a in attributes]
    for a in attributes:
        assert len(a.shape) == len(attributes[0].shape)

    if len(attributes[0].shape) == 1:
        return np.hstack(attributes)

    else:
        assert len(attributes[0].shape) == 2
        max_cols = max(a.shape[1] for a in attributes)
        padded_attributes = []
        for a in attributes:
            if a.shape[1] < max_cols:
                padding = np.empty((a.shape[0], max_cols-a.shape[1]), dtype=a.dtype)
                padding[:] = np.nan
                padded_attributes.append(np.hstack((a, padding)))
            else:
                padded_attributes.append(a)
        return np.vstack(padded_attributes)