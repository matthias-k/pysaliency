import json
from typing import Dict, List, Optional, Union

import numpy as np

from ..utils.variable_length_array import VariableLengthArray, concatenate_variable_length_arrays
from .utils import _load_attribute_dict_from_hdf5, create_hdf5_dataset, decode_string, get_merged_attribute_list, hdf5_wrapper


class Scanpaths(object):
    """
    Represents a collection of scanpaths.

    Attributes:
        xs (VariableLengthArray): The x-coordinates of the scanpaths.
        ys (VariableLengthArray): The y-coordinates of the scanpaths.
        n (np.ndarray): The image index
        length (np.ndarray): The length of each scanpath.
        scanpath_attributes (dict): Additional attributes associated with the scanpaths.
        fixation_attributes (dict): Additional attributes associated with the fixations in the scanpaths.
        attribute_mapping (dict): Mapping of attribute names to their corresponding values, will be used when creating `Fixations` instances from the `Scanpaths` instance.
             for example {'durations': 'duration'}
    """

    xs: VariableLengthArray
    ys: VariableLengthArray
    n: np.ndarray

    def __init__(self,
                 xs: Union[np.ndarray, VariableLengthArray],
                 ys: Union[np.ndarray, VariableLengthArray],
                 n: np.ndarray,
                 length=None,
                 scanpath_attributes: Optional[Dict[str, np.ndarray]] = None,
                 fixation_attributes: Optional[Dict[str, Union[np.ndarray, VariableLengthArray]]]=None,
                 attribute_mapping: Optional[Dict[str, str]] = None,
                 **kwargs):

        self.n = np.asarray(n)

        if not isinstance(xs, VariableLengthArray):
            self.xs = VariableLengthArray(xs, length)
        else:
            self.xs = xs

        if length is not None:
            if not np.all(self.xs.lengths == length):
                raise ValueError("Lengths of xs and lengths do not match")

        self.length = self.xs.lengths.copy()

        self.ys = self._as_variable_length_array(ys)

        if not len(self.xs) == len(self.ys) == len(self.n):
            raise ValueError("Length of xs, ys, ts and n has to match")

        scanpath_attributes = scanpath_attributes or {}
        fixation_attributes = fixation_attributes or {}
        self.attribute_mapping = attribute_mapping or {}
        self.attribute_mapping = dict(self.attribute_mapping)


        for key, value in kwargs.items():
            if value is None:
                continue
            if not len(value) == len(self.xs):
                raise ValueError(f"Length of attribute {key} has to match number of scanpaths, but got {len(value)} != {len(self.xs)}")
            if isinstance(value, VariableLengthArray) or isinstance(value[0], (list, np.ndarray)):
                if key in fixation_attributes:
                    raise ValueError(f"Attribute {key} already exists in fixation_attributes")
                fixation_attributes[key] = self._as_variable_length_array(value)
                if key not in self.attribute_mapping and key[-1] == 's':
                    self.attribute_mapping[key] = key[:-1]
            else:
                if key in scanpath_attributes:
                    raise ValueError(f"Attribute {key} already exists in scanpath_attributes")
                scanpath_attributes[key] = np.array(value)

        # setting scanpath attributes

        self.scanpath_attributes = {key: np.array(value) for key, value in scanpath_attributes.items()}

        for key, value in self.scanpath_attributes.items():
            if not len(value) == len(self.xs):
                raise ValueError(f"Length of scanpath attribute {key} has to match number of scanpaths, but got {len(value)} != {len(self.xs)}")

        # setting fixation attributes

        self.fixation_attributes = {key: self._as_variable_length_array(value) for key, value in fixation_attributes.items()}

    def _check_lengths(self, other: VariableLengthArray):
        if not len(self) == len(other):
            raise ValueError("Length of scanpaths has to match")
        if not np.all(self.length == other.lengths):
            raise ValueError("Lengths of scanpaths have to match")

    def _as_variable_length_array(self, data: Union[np.ndarray, VariableLengthArray]) -> VariableLengthArray:
        if not isinstance(data, VariableLengthArray):
            data = VariableLengthArray(data, self.length)

        self._check_lengths(data)

        return data

    def __len__(self):
        return len(self.xs)

    @property
    def ts(self) -> VariableLengthArray:
        return self.fixation_attributes['ts']

    @property
    def subject(self) -> VariableLengthArray:
        return self.scanpath_attributes['subject']


    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target):
        """ Write scanpaths to hdf5 file or hdf5 group
        """
        target.attrs['type'] = np.bytes_('Scanpaths')
        target.attrs['version'] = np.bytes_('1.0')

        target.create_dataset('xs', data=self.xs._data)
        target.create_dataset('ys', data=self.ys._data)
        target.create_dataset('n', data=self.n)
        target.create_dataset('length', data=self.length)

        scanpath_attributes_group = target.create_group('scanpath_attributes')
        for attribute_name, attribute_value in self.scanpath_attributes.items():
            create_hdf5_dataset(scanpath_attributes_group, attribute_name, attribute_value)
        scanpath_attributes_group.attrs['__attributes__'] = np.bytes_(json.dumps(sorted(self.scanpath_attributes.keys())))

        fixation_attributes_group = target.create_group('fixation_attributes')
        for attribute_name, attribute_value in self.fixation_attributes.items():
            fixation_attributes_group.create_dataset(attribute_name, data=attribute_value._data)
        fixation_attributes_group.attrs['__attributes__'] = np.bytes_(json.dumps(sorted(self.fixation_attributes.keys())))

        target.attrs['attribute_mapping'] = np.bytes_(json.dumps(self.attribute_mapping))


    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'Scanpaths':
            raise ValueError("Invalid type! Expected 'Scanpaths', got", data_type)

        valid_versions = ['1.0']
        if data_version not in valid_versions:
            raise ValueError("Invalid version! Expected one of {}, got {}".format(', '.join(valid_versions), data_version))

        length = source['length'][...]
        xs = VariableLengthArray(source['xs'][...], length)
        ys = VariableLengthArray(source['ys'][...], length)
        n = source['n'][...]

        scanpath_attributes = _load_attribute_dict_from_hdf5(source['scanpath_attributes'])

        fixation_attributes_group = source['fixation_attributes']
        json_attributes = fixation_attributes_group.attrs['__attributes__']
        if not isinstance(json_attributes, str):
            json_attributes = json_attributes.decode('utf8')
        __attributes__ = json.loads(json_attributes)

        fixation_attributes = {attribute: VariableLengthArray(fixation_attributes_group[attribute][...], length) for attribute in __attributes__}

        return cls(
            xs=xs,
            ys=ys,
            n=n,
            length=length,
            scanpath_attributes=scanpath_attributes,
            fixation_attributes=fixation_attributes,
            attribute_mapping=json.loads(decode_string(source.attrs['attribute_mapping']))
        )

    def __getitem__(self, index):
        # TODO
        # - integer to return single scanpath
        # - 2d index to return single Fixation (for now via index of scanpath and index of fixation in scanpath)
        # - 2d index array to return Fixations instance (for now via index of scanpath and index of fixation in scanpath)

        if isinstance(index, tuple):
            raise NotImplementedError("Not implemented yet")
        elif isinstance(index, int):
            raise NotImplementedError("Not implemented yet")
        else:
            return type(self)(self.xs[index], self.ys[index], self.n[index], self.length[index],
                              scanpath_attributes={key: value[index] for key, value in self.scanpath_attributes.items()},
                              fixation_attributes={key: value[index] for key, value in self.fixation_attributes.items()},
                              attribute_mapping=self.attribute_mapping)

    def copy(self) -> 'Scanpaths':
        return type(self)(self.xs.copy(), self.ys.copy(), self.n.copy(), self.length.copy(),
                          scanpath_attributes={key: value.copy() for key, value in self.scanpath_attributes.items()},
                          fixation_attributes={key: value.copy() for key, value in self.fixation_attributes.items()},
                          attribute_mapping=self.attribute_mapping.copy())

    @classmethod
    def concatenate(cls, scanpaths_list: List['Scanpaths']) -> 'Scanpaths':
        return concatenate_scanpaths(scanpaths_list)


def concatenate_scanpaths(scanpaths_list: List[Scanpaths]) -> Scanpaths:
    xs = concatenate_variable_length_arrays([scanpaths.xs for scanpaths in scanpaths_list])
    ys = concatenate_variable_length_arrays([scanpaths.ys for scanpaths in scanpaths_list])
    n = np.concatenate([scanpaths.n for scanpaths in scanpaths_list])
    length = np.concatenate([scanpaths.length for scanpaths in scanpaths_list])

    merged_scanpath_attributes = get_merged_attribute_list([scanpaths.scanpath_attributes.keys() for scanpaths in scanpaths_list])
    scanpath_attributes = {key: np.concatenate([scanpaths.scanpath_attributes[key] for scanpaths in scanpaths_list]) for key in merged_scanpath_attributes}

    merged_fixation_attributes = get_merged_attribute_list([scanpaths.fixation_attributes.keys() for scanpaths in scanpaths_list])
    fixation_attributes = {key: concatenate_variable_length_arrays([scanpaths.fixation_attributes[key] for scanpaths in scanpaths_list]) for key in merged_fixation_attributes}

    merged_attribute_mapping = {}
    for key in merged_fixation_attributes:
        mappings = {scanpaths.attribute_mapping.get(key) for scanpaths in scanpaths_list}
        if len(mappings) > 1:
            raise ValueError(f"Multiple mappings for attribute {key} found: {mappings}")
        elif len(mappings) == 1 and list(mappings)[0] is not None:
            merged_attribute_mapping[key] = mappings.pop()

    return Scanpaths(xs, ys, n, length, scanpath_attributes=scanpath_attributes, fixation_attributes=fixation_attributes, attribute_mapping=merged_attribute_mapping)