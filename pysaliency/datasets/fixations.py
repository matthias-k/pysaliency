import json
import warnings
from typing import Dict, List, Optional, Tuple, Union

import deprecation
import numpy as np
from tqdm import tqdm

from ..utils import remove_trailing_nans, deprecated_class
from ..utils.variable_length_array import VariableLengthArray
from .scanpaths import Scanpaths
from .utils import _load_attribute_dict_from_hdf5, concatenate_attributes, decode_string, get_merged_attribute_list, hdf5_wrapper


class Fixations(object):
    """Capsules the fixations of a dataset and provides different methods
       of accessing them, e.g. in fixation trains, as conditional fixations
       or just all fixations at once.

       Fixations consist of:
           x: the x-position of the fixation
           y: the y-position of the fixation
           t: the time of the fixation
           x_hist: the previous x-positions in the history of this fixation
           y_hist: the previous y-positions in the history of this fixation
           t_hist: the previous times in the history of this fixation
           subject: the subject who made the fixation
           n: the number of the stimuli (optional, only needed when evaluating not on single images)

        Fixations support slicing via fixations[indices] as a shortcut for fixations.filter.

        Although all fixations have a history of previous fixations, these histories
        do not have to form a set of fixation sequences. For example, if a fixation
        has a previous fixation, this previous fixation does not have to be as a
        fixation of its on in the dataset. This is important because otherwise
        a lot of useful filtering operations would not be possible (e.g. filter
        for all fixations with at least one previous fixation to calculate
        saccade lengths). If you need fixation trains, use the subclass
        `FixationTrains`.
    """
    def __init__(self,
                 x: Union[List, np.ndarray],
                 y: Union[List, np.ndarray],
                 t: Union[List, np.ndarray],
                 x_hist: Union[List, VariableLengthArray],
                 y_hist: Union[List, VariableLengthArray],
                 t_hist: Union[List, VariableLengthArray],
                 n: Union[List, np.ndarray],
                 subject: Optional[Union[List, np.ndarray]] = None,
                 subjects: Optional[Union[List, np.ndarray]] = None,
                 attributes: Optional[Dict[str, Union[np.ndarray, VariableLengthArray]]] = None):

        self.__attributes__ = []

        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.t = np.asarray(t)
        self.n = np.asarray(n)

        # would be nice, is not yet supported. But we can simply pass the VariableLengthArray instead
        # if isinstance(x_hist, list):
        #     x_hist = VariableLengthArray(x_hist)
        #     self.lengths = x_hist.lengths
        if isinstance(x_hist, (list, np.ndarray)):
            x_hist = np.array(x_hist)
            self.scanpath_history_length = (1 - np.isnan(x_hist)).sum(axis=-1)
            x_hist = VariableLengthArray(x_hist, lengths=self.scanpath_history_length)
        elif isinstance(x_hist, VariableLengthArray):
            self.scanpath_history_length = x_hist.lengths


        y_hist = self._as_variable_length_array(y_hist)
        t_hist = self._as_variable_length_array(t_hist)

        if subjects is not None:
            warnings.warn("subjects is deprecated, use subject instead", DeprecationWarning, stacklevel=2)
            subject = subjects

        if subject is not None:
            self.__attributes__.append('subject')
            subject = np.asarray(subject)

        self.x_hist = x_hist
        self.y_hist = y_hist
        self.t_hist = t_hist
        self.subject = subject

        if not len(self.x) == len(self.y) == len(self.t) == len(self.x_hist) == len(self.y_hist) == len(self.t_hist) == len(self.n):
            raise ValueError("Lengths of fixations have to match")
        if self.subject is not None and not len(self.x) == len(self.subject):
            raise ValueError("Length of subject has to match number of fixations")

        if attributes is not None:
            for name, value in attributes.items():
                if name not in self.__attributes__:
                    self.__attributes__.append(name)
                if not len(value) == len(self.x):
                    raise ValueError(f"Length of attribute '{name}' has to match number of fixations")
                setattr(self, name, value)


    def _check_lengths(self, other: VariableLengthArray):
        if not len(self) == len(other):
            raise ValueError("Length of scanpaths has to match")
        if not np.all(self.scanpath_history_length == other.lengths):
            raise ValueError("Lengths of scanpaths have to match")

    def _as_variable_length_array(self, data: Union[np.ndarray, VariableLengthArray]) -> VariableLengthArray:
        if not isinstance(data, VariableLengthArray):
            data = VariableLengthArray(data, self.scanpath_history_length)

        self._check_lengths(data)

        return data

    @property
    @deprecation.deprecated(deprecated_in="0.3.0", removed_in="1.0.0", details="Use `scanpath_history_length` instead")
    def lengths(self):
        return self.scanpath_history_length

    @property
    @deprecation.deprecated(deprecated_in="0.3.0", removed_in="1.0.0", details="Use `subject` instead")
    def subjects(self):
        return self.subject

    @classmethod
    @deprecation.deprecated(deprecated_in="0.3.0", removed_in="1.0.0", details="Use `FixationsWithoutHistory` instead")
    def create_without_history(cls, x, y, n, subjects=None):
        """ Create new fixation object from fixation data without time and optionally
            without subject information
        """
        if subjects is None:
            subjects = np.ones(len(x))
        return cls.FixationsWithoutHistory(x, y, np.zeros(len(x)), n, subjects)

    @classmethod
    def from_fixation_matrices(cls, matrices):
        """
        create new Fixation object with fixations from fixation matrices

        Often, fixations are stored in fixation matrices: For each stimulus, there
        is a matrix of the same size as the image which ones in each fixated location
        and zero everywhere else. This method allows to construct a `Fixation` instance
        from such fixation matrices.

        >>> matrix1 = np.zeros((10,10))
        >>> matrix1[5, 2] = 1
        >>> matrix1[3, 3] = 1
        >>> matrix2  = np.zeros((20, 30))
        >>> matrix2[10, 20] = 1
        >>> fixations = pysaliency.Fixation.from_fixation_matrices(
                [matrix1,
                 matrix2])
        """
        xs = []
        ys = []
        ns = []
        for _n, matrix in enumerate(matrices):
            y, x = np.nonzero(matrix)
            n = [_n] * len(y)
            xs.append(x)
            ys.append(y)
            ns.append(n)
        x = np.hstack(xs).astype(float)
        y = np.hstack(ys).astype(float)
        n = np.hstack(ns)
        return cls.create_without_history(x, y, n)

    @classmethod
    def concatenate(cls, fixations):
        kwargs = {}
        for key in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n', 'subject']:
            kwargs[key] = concatenate_attributes(getattr(f, key) for f in fixations)

        attributes = get_merged_attribute_list([f.__attributes__ for f in fixations])
        attribute_dict = {}
        for key in attributes:
            if key == 'subject':
                continue
            attribute_dict[key] = concatenate_attributes(getattr(f, key) for f in fixations)

        kwargs['attributes'] = attribute_dict

        new_fixations = cls(**kwargs)

        return new_fixations

    def __getitem__(self, indices):
        return self.filter(indices)

    def __len__(self):
        return len(self.x)

    def filter(self, inds):
        """
        Create new fixations object which contains only the fixations with indexes in inds

        .. note::
            The fixation trains of the object are left as is. Filtering the fixation trains
            is not possible as the indices may include only some fixation of a fixation train.

            The attributes `consistent_fixation_trains` tracks whether a `Fixations` instance
            still has consistent fixation trains. The return of this function will be marked
            to have inconsistent fixation trains. If you need to filter with consistent
            fixation trains, use `Fixations.filter_fixation_trains`.
        """

        kwargs = {}
        other_attributes = {}

        def filter_array(name):
            kwargs[name] = getattr(self, name)[inds].copy()

        for name in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n']:
            filter_array(name)

        for name in self.__attributes__:
            filter_array(name)
            if name != 'subject':
                other_attributes[name] = kwargs.pop(name)

        kwargs['attributes'] = other_attributes

        return Fixations(**kwargs)

    def _get_previous_values(self, name, index):
        """return fixations.name[np.arange(len(fixations.name)),index]"""
        a = getattr(self, name)
        inds = np.arange(len(a))
        if index >= 0:
            return a[inds, index]
        else:
            indexes = self.scanpath_history_length + index
            return a[inds, indexes]

    def get_saccade(self, index = -1):
        """
        Return saccades for all fixations.

        @type  index: integer
        @param index: index of the saccade to return. `index==-1` returns the
                      the last saccades of all fixations etc.

        @return dx, dy, dt of the saccade

        Example:

            dx, dy, dt = fixations.get_saccade(-1)
            mean_saccade_length = np.sqrt(dx**2+dy**2).mean()
        """

        if index > 0:
            raise NotImplemented()
        if index == -1:
            x1 = self.x
            y1 = self.y
            t1 = self.t
        else:
            x1 = self._get_previous_values('x_hist', index+1)
            y1 = self._get_previous_values('y_hist', index+1)
            t1 = self._get_previous_values('t_hist', index+1)
        dx = x1 - self._get_previous_values('x_hist', index)
        dy = y1 - self._get_previous_values('y_hist', index)
        dt = t1 - self._get_previous_values('t_hist', index)
        return dx, dy, dt
        #return np.vstack((dy,dx)).T

    @property
    def x_int(self):
        """ x coordinates of the fixations, converted to integers """
        return np.asarray(self.x, dtype=int)

    @property
    def y_int(self):
        """ y coordinates of the fixations, converted to integers """
        return np.asarray(self.y, dtype=int)

    @property
    def subject_count(self):
        return int(self.subject.max())+1

    def copy(self):
        cfix = Fixations(self.x.copy(), self.y.copy(), self.t.copy(),
                         self.x_hist.copy(), self.y_hist.copy(), self.t_hist.copy(),
                         self.n.copy(), self.subject.copy() if self.subject is not None else None)
        cfix.__attributes__ = list(self.__attributes__)
        for name in self.__attributes__:
            setattr(cfix, name, getattr(self, name).copy())
        return cfix

    @classmethod
    def FixationsWithoutHistory(cls, x, y, t, n, subject=None, subjects=None):

        if subjects is not None:
            warnings.warn("subjects is deprecated, use subject instead", DeprecationWarning, stacklevel=2)
            subject = subjects

        x_hist = np.empty((len(x), 1))
        x_hist[:] = np.nan
        y_hist = np.empty((len(x), 1))
        y_hist[:] = np.nan
        t_hist = np.empty((len(x), 1))
        t_hist[:] = np.nan
        return cls(x, y, t, x_hist, y_hist, t_hist, n, subject)

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target):
        """ Write fixations to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.bytes_('Fixations')
        target.attrs['version'] = np.bytes_('1.2')

        variable_length_arrays = []

        for attribute in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n', 'lengths'] + self.__attributes__:
            if attribute == 'lengths':
                data = self.scanpath_history_length
            else:
                data = getattr(self, attribute)
            if isinstance(data, VariableLengthArray):
                variable_length_arrays.append(attribute)
                data = data._data
            target.create_dataset(attribute, data=data)

        target.attrs['__attributes__'] = np.bytes_(json.dumps(self.__attributes__))
        target.attrs['__variable_length_arrays__'] = np.bytes_(json.dumps(sorted(variable_length_arrays)))

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        """ Read fixations from hdf5 file or hdf5 group """

        # TODO: rewrite to use constructor instead of manipulating the object directly

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'Fixations':
            raise ValueError("Invalid type! Expected 'Fixations', got", data_type)

        if data_version not in ['1.0', '1.1', '1.2']:
            raise ValueError("Invalid version! Expected '1.0', '1.1' or '1.2', got", data_version)

        if data_version < '1.2':
            data = {key: source[key][...] for key in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n', 'subjects']}
            data['subject'] = data['subjects']
            del data['subjects']
        else:
            data = {key: source[key][...] for key in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n', 'subject']}
        fixations = cls(**data)

        json_attributes = source.attrs['__attributes__']
        if not isinstance(json_attributes, str):
            json_attributes = json_attributes.decode('utf8')
        __attributes__ = json.loads(json_attributes)
        fixations.__attributes__ = list(__attributes__)

        if data_version >= '1.1':
            lengths = source['lengths'][...]

            json_variable_length_arrays = source.attrs['__variable_length_arrays__']
            if not isinstance(json_variable_length_arrays, str):
                json_variable_length_arrays = json_variable_length_arrays.decode('utf8')
            variable_length_arrays = json.loads(json_variable_length_arrays)

        else:
            lengths = fixations.scanpath_history_length
            variable_length_arrays = ['x_hist', 'y_hist', 't_hist'] + [key for key in __attributes__ if key.endswith('_hist')]

        for key in __attributes__:
            data = source[key][...]

            if key in variable_length_arrays:
                data = VariableLengthArray(data, lengths)

            if key == 'subjects' and data_version < '1.2':
                key = 'subject'
                fixations.__attributes__[fixations.__attributes__.index('subjects')] = 'subject'

            setattr(fixations, key, data)

        return fixations


class ScanpathFixations(Fixations):
    """
    Contains fixations which come from full scanpaths (as opposed to potentially only some fixations of a scanpath).
    """
    def __init__(self, scanpaths: Scanpaths):
        N_fixations = scanpaths.length.sum()
        max_scanpath_length = scanpaths.length.max() if len(scanpaths) else 0
        max_history_length = max(max_scanpath_length - 1, 0)

        # Create conditional fixations
        x = np.empty(N_fixations)
        y = np.empty(N_fixations)
        t = np.empty(N_fixations)

        x_hist = []
        y_hist = []
        t_hist = []
        n = np.empty(N_fixations, dtype=int)
        subject = np.empty(N_fixations, dtype=int)
        scanpath_index = np.empty(N_fixations, dtype=int)

        out_index = 0
        # TODO: maybe implement in numba?
        # probably best: have function fill_fixation_data(scanpath_data, fixation_data, hist_data=None)
        for train_index in range(len(scanpaths)):
            for fix_index in range(scanpaths.length[train_index]):
                x[out_index] = scanpaths.xs[train_index][fix_index]
                y[out_index] = scanpaths.ys[train_index][fix_index]
                t[out_index] = scanpaths.ts[train_index][fix_index]
                n[out_index] = scanpaths.n[train_index]
                # subject[out_index] = scanpaths.scanpath_attributes['subject'][train_index]
                scanpath_index[out_index] = train_index
                x_hist.append(scanpaths.xs[train_index][:fix_index])
                y_hist.append(scanpaths.ys[train_index][:fix_index])
                t_hist.append(scanpaths.ts[train_index][:fix_index])
                out_index += 1

        x_hist = VariableLengthArray(x_hist)
        y_hist = VariableLengthArray(y_hist)
        t_hist = VariableLengthArray(t_hist)

        auto_attributes = []
        attributes = {
            'scanpath_index': scanpath_index,
        }

        for attribute_name, value in scanpaths.scanpath_attributes.items():
            new_attribute_name = scanpaths.attribute_mapping.get(attribute_name, attribute_name)
            if new_attribute_name in attributes:
                raise ValueError("attribute name clash: {new_attribute_name}".format(new_attribute_name=new_attribute_name))
            attribute_shape = [] if not len(value) else np.asarray(value[0]).shape
            attributes[new_attribute_name] = np.empty([N_fixations] + list(attribute_shape), dtype=value.dtype)
            auto_attributes.append(new_attribute_name)

            out_index = 0
            for train_index in range(len(scanpaths)):
                for _ in range(scanpaths.length[train_index]):
                    attributes[new_attribute_name][out_index] = value[train_index]
                    out_index += 1


        for attribute_name, value in scanpaths.fixation_attributes.items():
            if attribute_name == 'ts':
                continue
            new_attribute_name = scanpaths.attribute_mapping.get(attribute_name, attribute_name)
            if new_attribute_name in attributes:
                raise ValueError("attribute name clash: {new_attribute_name}".format(new_attribute_name=new_attribute_name))
            attributes[new_attribute_name] = np.empty(N_fixations)
            auto_attributes.append(new_attribute_name)

            hist_attribute_name = new_attribute_name + '_hist'
            if hist_attribute_name in attributes:
                raise ValueError("attribute name clash: {hist_attribute_name}".format(hist_attribute_name=hist_attribute_name))
            attributes[hist_attribute_name] = np.full((N_fixations, max_history_length), fill_value=np.nan)
            auto_attributes.append(hist_attribute_name)

            out_index = 0
            for train_index in range(len(scanpaths)):
                for fix_index in range(scanpaths.length[train_index]):
                    attributes[new_attribute_name][out_index] = value[train_index, fix_index]
                    attributes[hist_attribute_name][out_index][:fix_index] = value[train_index, :fix_index]
                    out_index += 1

            attributes[hist_attribute_name] = VariableLengthArray(attributes[hist_attribute_name], x_hist.lengths)

        super().__init__(
            x=x,
            y=y,
            t=t,
            x_hist=x_hist,
            y_hist=y_hist,
            t_hist=t_hist,
            n=n,
            subject=subject,
            attributes=attributes
        )

        self.scanpaths = scanpaths
        self.auto_attributes = auto_attributes


    def copy(self) -> 'ScanpathFixations':
        copied_scanpaths = self.scanpaths.copy()
        return ScanpathFixations(scanpaths=copied_scanpaths)

    @classmethod
    def concatenate(cls, scanpath_fixations: List['ScanpathFixations']) -> 'ScanpathFixations':
        concatenated_scanpaths = Scanpaths.concatenate([sf.scanpaths for sf in scanpath_fixations])
        return ScanpathFixations(scanpaths=concatenated_scanpaths)

    def filter_scanpaths(self, indices) -> 'ScanpathFixations':
        filtered_scanpaths = self.scanpaths[indices]
        return ScanpathFixations(scanpaths=filtered_scanpaths)

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target):
        """ Write ScanpathFixations to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.bytes_('ScanpathFixations')
        target.attrs['version'] = np.bytes_('1.0')

        self.scanpaths.to_hdf5(target.create_group('scanpaths'))

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        """ Read ScanpathFixations from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type == 'FixationTrains':
            fixation_trains = FixationTrains.read_hdf5(source)
            if fixation_trains.non_auto_attributes:
                print("NA", fixation_trains.non_auto_attributes)
                non_auto_attributes = ', '.join(fixation_trains.non_auto_attributes)
                raise ValueError(f"FixationTrains object has non-auto attributes ({non_auto_attributes}), can't convert to ScanpathFixations")
            return cls(scanpaths=fixation_trains.scanpaths)

        if data_type != 'ScanpathFixations':
            raise ValueError("Invalid type! Expected 'ScanpathFixations', got", data_type)

        if data_version != '1.0':
            raise ValueError("Invalid version! Expected '1.0', got", data_version)

        scanpaths = Scanpaths.read_hdf5(source['scanpaths'])

        return cls(scanpaths=scanpaths)


@deprecated_class(deprecated_in="0.3.0", removed_in="1.0.0", details="Use `ScanpathFixations` instead")
class FixationTrains(ScanpathFixations):
    """
    Capsules the fixations of a dataset as fixation trains.

    Additionally to `Fixations`, `FixationTrains`-instances
    have the attributes
        train_xs: 2d array (number_of_trains, maximum_length_of_train)
        train_ys: 2d array (number_of_trains, maximum_length_of_train)
        train_ts: 2d array (number_of_trains, maximum_length_of_train)
        train_ns: 1d array (number_of_trains)
        train_subjects: 1d array (number_of_trains)

    scanpath_attributes: dictionary of attributes applying to full scanpaths, e.g. task
            {attribute_name: $NUM_SCANPATHS-length-list}
            scanpath attributes will automatically also become attributes
    scanpath_fixation_attribute: dictionary of attributes applying to fixations in the scanpath, e.g. duration
            {attribute_name: $NUM_SCANPATH x $NUM_FIXATIONS_IN_SCANPATH}
            scanpath fixation attributes will generate two attributes: the value for each fixation
            and the history for previous fixations. E.g. a scanpath fixation attribute "durations" will generate
            an attribute "durations" and an attribute "durations_hist"

    """
    def __init__(self, train_xs, train_ys, train_ts, train_ns, train_subjects, scanpath_attributes=None, scanpath_fixation_attributes=None, attributes=None, scanpath_attribute_mapping=None, scanpaths=None):

        # raise ValueError("DON'T USE FIXATIONTRAINS ANYMORE, USE SCANPATHFIXATIONS INSTEAD")

        if isinstance(train_xs, Scanpaths):
            scanpaths = train_xs

        elif scanpaths is None:
            scanpath_attributes = scanpath_attributes or {}
            scanpath_attribute_mapping = scanpath_attribute_mapping or {}
            scanpath_fixation_attributes = scanpath_fixation_attributes or {}

            lengths = [len(remove_trailing_nans(xs)) for xs in train_xs]

            scanpaths = Scanpaths(
                xs=train_xs,
                ys=train_ys,
                ts=train_ts,
                n=train_ns,
                subject=train_subjects,
                length=lengths,
                scanpath_attributes=scanpath_attributes,
                fixation_attributes=scanpath_fixation_attributes,
                attribute_mapping=scanpath_attribute_mapping,
            )

        super().__init__(scanpaths=scanpaths)

        if attributes is None:
            attributes = {}

        if attributes:
            warnings.warn("Don't use attributes for FixationTrains, use scanpath_attributes or scanpath_fixation_attributes instead! FixationTrains is deprecated, the successor ScanpathFixations doesn't support attributes anymore", stacklevel=2, category=DeprecationWarning)

        if attributes:
            self.__attributes__ = list(self.__attributes__)
            for name, value in attributes.items():
                if name not in self.__attributes__:
                    self.__attributes__.append(name)
                else:
                    raise ValueError(f"attribute {name} already set!")
                if not len(value) == len(self):
                    raise ValueError(f"Length of attribute '{name}' has to match number of fixations")
                setattr(self, name, value)

        self.full_nonfixations = None

    @classmethod
    def from_scanpaths(cls, scanpaths: Scanpaths, attributes: Optional[Dict]=None):
        return cls(
            train_xs=None,
            train_ys=None,
            train_ts=None,
            train_ns=None,
            train_subjects=None,
            attributes=attributes,
            scanpaths=scanpaths
        )

    @property
    def train_xs(self) -> VariableLengthArray:
        return self.scanpaths.xs

    @property
    def train_ys(self) -> VariableLengthArray:
        return self.scanpaths.ys

    @property
    def train_ts(self) -> VariableLengthArray:
        return self.scanpaths.ts

    @property
    def train_ns(self) -> np.ndarray:
        return self.scanpaths.n

    @property
    def train_subjects(self) -> VariableLengthArray:
        return self.scanpaths.subject

    @property
    def train_lengths(self) -> np.ndarray:
        return self.scanpaths.length

    @property
    def scanpath_attributes(self) -> Dict[str, np.ndarray]:
        return {
            key: value for key, value in self.scanpaths.scanpath_attributes.items() if key != 'subject'
        }

    @property
    def scanpath_fixation_attributes(self) -> Dict[str, VariableLengthArray]:
        return {
            key: value for key, value in self.scanpaths.fixation_attributes.items() if key != 'ts'
        }

    @property
    def scanpath_attribute_mapping(self) -> Dict[str, str]:
        return {
            key: value for key, value in self.scanpaths.attribute_mapping.items() if key != 'ts'
        }

    @property
    def non_auto_attributes(self):
        """lists all attributes of this `FixationTrains` instance which are not auto generated from scanpath attributes"""
        return [attribute_name for attribute_name in self.__attributes__ if attribute_name not in self.auto_attributes + ['scanpath_index']]

    @classmethod
    def concatenate(cls, fixation_trains: List['FixationTrains']) -> 'FixationTrains':

        concatenated_scanpaths = Scanpaths.concatenate([f.scanpaths for f in fixation_trains])

        attributes = get_merged_attribute_list([f.non_auto_attributes for f in fixation_trains])
        attribute_dict = {}
        for key in attributes:
            attribute_dict[key] = concatenate_attributes(getattr(f, key) for f in fixation_trains)

        return cls.from_scanpaths(concatenated_scanpaths, attributes=attribute_dict)


    def set_scanpath_attribute(self, name, data, fixation_attribute_name=None):
        """Sets a scanpath attribute
        name: name of scanpath attribute
        data: data of scanpath attribute, has to be of same length as number of scanpaths
        fixation_attribute: name of automatically generated fixation attribute if it should be different than scanpath attribute name
        """
        if not len(data) == len(self.train_xs):
            raise ValueError(f'Length of scanpath attribute data has to match number of scanpaths: {len(data)} != {len(self.train_xs)}')
        self.scanpath_attributes[name] = data

        if fixation_attribute_name is not None:
            self.scanpath_attribute_mapping[name] = fixation_attribute_name

        new_attribute_name = self.scanpath_attribute_mapping.get(name, name)
        if new_attribute_name in self.attributes and new_attribute_name not in self.auto_attributes:
            raise ValueError("attribute name clash: {new_attribute_name}".format(new_attribute_name=new_attribute_name))

        attribute_shape = np.asarray(data[0]).shape
        self.attributes[new_attribute_name] = np.empty([len(self.train_xs)] + list(attribute_shape), dtype=data.dtype)
        if new_attribute_name not in self.auto_attributes:
            self.auto_attributes.append(new_attribute_name)

        out_index = 0
        for train_index in range(len(self.train_xs)):
            fix_length = (1 - np.isnan(self.train_xs[train_index])).sum()
            for _ in range(fix_length):
                self.attributes[new_attribute_name][out_index] = self.scanpath_attributes[name][train_index]
                out_index += 1

    def copy(self):
        copied_attributes = {}
        for attribute_name in self.__attributes__:
            if attribute_name in ['scanpath_index'] + self.auto_attributes:
                continue
            copied_attributes[attribute_name] = getattr(self, attribute_name).copy()
        copied_scanpaths = FixationTrains(
            train_xs=self.train_xs.copy(),
            train_ys=self.train_ys.copy(),
            train_ts=self.train_ts.copy(),
            train_ns=self.train_ns.copy(),
            train_subjects=self.train_subjects.copy(),
            scanpath_attributes={
                key: value.copy() for key, value in self.scanpath_attributes.items()
            } if self.scanpath_attributes else None,
            scanpath_fixation_attributes={
                key: value.copy() for key, value in self.scanpath_fixation_attributes.items()
            } if self.scanpath_fixation_attributes else None,
            scanpath_attribute_mapping=dict(self.scanpath_attribute_mapping),
            attributes=copied_attributes if copied_attributes else None,
        )
        return copied_scanpaths

    def filter_scanpaths(self, indices):
        """
        Create new fixations object which contains only the scanpaths indicated.
        """

        filtered_scanpaths = self.scanpaths[indices]

        scanpath_indices = np.arange(len(self.scanpaths), dtype=int)[indices]
        fixation_indices = np.in1d(self.scanpath_index, scanpath_indices)

        attributes = {
            attribute_name: getattr(self, attribute_name)[fixation_indices] for attribute_name in self.__attributes__ if attribute_name not in ['scanpath_index'] + self.auto_attributes
        }

        return type(self).from_scanpaths(
            scanpaths=filtered_scanpaths,
            attributes=attributes
        )

    @deprecation.deprecated(deprecated_in="0.3.0", removed_in="1.0.0", details="Use `FixationTrains.filter_scanpaths` instead")
    def filter_fixation_trains(self, indices):
        """
        Create new fixations object which contains only the scanpaths indicated.
        """

        return self.filter_scanpaths(indices)

    def fixation_trains(self):
        """Yield for every fixation train of the dataset:
             xs, ys, ts, n, subject
        """
        for i in range(len(self.train_xs)):
            length = (1 - np.isnan(self.train_xs[i])).sum()
            xs = self.train_xs[i][:length]
            ys = self.train_ys[i][:length]
            ts = self.train_ts[i][:length]
            n = self.train_ns[i]
            subject = self.train_subjects[i]
            yield xs, ys, ts, n, subject

    @classmethod
    def from_fixation_trains(cls, xs, ys, ts, ns, subject=None, subjects=None, attributes=None, scanpath_attributes=None, scanpath_fixation_attributes=None, scanpath_attribute_mapping=None):
        """ Create Fixation object from fixation trains.
              - xs, ys, ts: Lists of array_like of double. Each array has to contain
                    the data from one fixation train.
              - ns, subjects: lists of int. ns has to contain the image index for
                    each fixation train, subjects the subject index for each
                    fixation train
        """
        if subjects is not None:
            warnings.warn("subjects is deprecated, use subject instead", DeprecationWarning, stacklevel=2)
            subject = subjects

        maxlength = max([len(x_train) for x_train in xs])
        train_xs = np.empty((len(xs), maxlength))
        train_xs[:] = np.nan
        train_ys = np.empty((len(xs), maxlength))
        train_ys[:] = np.nan
        train_ts = np.empty((len(xs), maxlength))
        train_ts[:] = np.nan
        train_ns = np.empty(len(xs), dtype=int)
        train_subjects = np.empty(len(xs), dtype=int)

        padded_scanpath_fixation_attributes = {}
        if scanpath_fixation_attributes is not None:
            for key, value in scanpath_fixation_attributes.items():
                assert len(value) == len(xs)
                if isinstance(value, list):
                    padded_scanpath_fixation_attributes[key] = np.full((len(xs), maxlength), fill_value=np.nan, dtype=float)

        for i in range(len(train_xs)):
            length = len(xs[i])
            train_xs[i, :length] = xs[i]
            train_ys[i, :length] = ys[i]
            train_ts[i, :length] = ts[i]
            train_ns[i] = ns[i]
            train_subjects[i] = subject[i]
            for attribute_name in padded_scanpath_fixation_attributes.keys():
                padded_scanpath_fixation_attributes[attribute_name][i, :length] = scanpath_fixation_attributes[attribute_name][i]

        return cls(
            train_xs,
            train_ys,
            train_ts,
            train_ns,
            train_subjects,
            attributes=attributes,
            scanpath_attributes=scanpath_attributes,
            scanpath_fixation_attributes=padded_scanpath_fixation_attributes,
            scanpath_attribute_mapping=scanpath_attribute_mapping)

    def generate_crossval(self, splitcount = 10):
        train_xs_training = []
        train_xs_eval = []
        train_ys_training = []
        train_ys_eval = []
        train_ts_training = []
        train_ts_eval = []
        train_ns_training = []
        train_ns_eval = []
        train_subjects_training = []
        train_subjects_eval = []
        # We have to make the crossvalidation data
        # reproducible. Therefore we use a
        # RandomState with fixed seed for the shuffling.
        rs = np.random.RandomState(42)
        for n in range(self.n.max()+1):
            inds = np.nonzero(self.train_ns == n)[0]
            rs.shuffle(inds)
            parts = np.array_split(inds, splitcount)
            for eval_index in range(splitcount):
                for index in range(splitcount):
                    part = parts[index]
                    if len(part) == 0:
                        continue
                    xs = self.train_xs[part]
                    ys = self.train_ys[part]
                    ts = self.train_ts[part]
                    ns = self.train_ns[part]
                    subjects = self.train_subjects[part]
                    if index == eval_index:
                        train_xs_eval.append(xs)
                        train_ys_eval.append(ys)
                        train_ts_eval.append(ts)
                        train_ns_eval.append(ns * splitcount + eval_index)
                        train_subjects_eval.append(subjects)
                    else:
                        train_xs_training.append(xs)
                        train_ys_training.append(ys)
                        train_ts_training.append(ts)
                        train_ns_training.append(ns * splitcount + eval_index)
                        train_subjects_training.append(subjects)
        train_xs_eval = np.vstack(train_xs_eval)
        train_ys_eval = np.vstack(train_ys_eval)
        train_ts_eval = np.vstack(train_ts_eval)
        train_ns_eval = np.hstack(train_ns_eval)
        train_subjects_eval = np.hstack(train_subjects_eval)
        train_xs_training = np.vstack(train_xs_training)
        train_ys_training = np.vstack(train_ys_training)
        train_ts_training = np.vstack(train_ts_training)
        train_ns_training = np.hstack(train_ns_training)
        train_subjects_training = np.hstack(train_subjects_training)
        fixations_training = type(self).from_fixation_trains(train_xs_training, train_ys_training,
                                                             train_ts_training, train_ns_training,
                                                             train_subjects_training)
        fixations_evaluation = type(self).from_fixation_trains(train_xs_eval, train_ys_eval,
                                                               train_ts_eval, train_ns_eval,
                                                               train_subjects_eval)
        return fixations_training, fixations_evaluation

    def shuffle_fixations(self, stimuli=None):
        new_indices = []
        new_ns = []
        if stimuli:
            widths = np.asarray([s[1] for s in stimuli.sizes]).astype(float)
            heights = np.asarray([s[0] for s in stimuli.sizes]).astype(float)
            x_factors = []
            y_factors = []
        for n in range(self.n.max()+1):
            inds = np.nonzero(~(self.n == n))[0]
            new_indices.extend(inds)
            new_ns.extend([n]*len(inds))
            if stimuli:
                other_ns = self.n[inds]
                x_factors.extend(stimuli.sizes[n][1]/widths[other_ns])
                y_factors.extend(stimuli.sizes[n][0]/heights[other_ns])
        new_fixations = self[new_indices]
        new_fixations.n = np.asarray(new_ns)
        if stimuli:
            x_factors = np.asarray(x_factors)
            y_factors = np.asarray(y_factors)
            new_fixations.x = x_factors*new_fixations.x
            new_fixations.x_hist = x_factors[:, np.newaxis]*new_fixations.x_hist
            new_fixations.y = y_factors*new_fixations.y
            new_fixations.y_hist = y_factors[:, np.newaxis]*new_fixations.y_hist
        return new_fixations

    def shuffle_fixation_trains(self, stimuli=None):
        """

        """
        if not self.consistent_fixation_trains:
            raise ValueError('Cannot shuffle fixation trains as fixation trains not consistent!')
        train_xs = []
        train_ys = []
        train_ts = []
        train_ns = []
        train_subjects = []
        for n in range(self.n.max()+1):
            inds = ~(self.train_ns == n)
            train_xs.append(self.train_xs[inds])
            train_ys.append(self.train_ys[inds])
            train_ts.append(self.train_ts[inds])
            train_ns.append(np.ones(inds.sum(), dtype=int)*n)
            train_subjects.append(self.train_subjects[inds])
        train_xs = np.vstack(train_xs)
        train_ys = np.vstack(train_ys)
        train_ts = np.vstack(train_ts)
        train_ns = np.hstack(train_ns)
        train_subjects = np.hstack(train_subjects)
        full_nonfixations = type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)
        #self.full_nonfixations = full_nonfixations
        return full_nonfixations

    def generate_full_nonfixations(self, stimuli=None):
        """
        Generate nonfixational distribution from this
        fixation object by using all fixation trains of
        other images. The individual fixation trains
        will be left intact.

        .. warning::
            This function operates on the fixation trains.
            Therefore, for filtered fixation objects it
            might return wrong results.
        """
        if self.full_nonfixations is not None:
            print("Reusing nonfixations!")
            return self.full_nonfixations
        train_xs = []
        train_ys = []
        train_ts = []
        train_ns = []
        train_subjects = []
        for n in range(self.n.max()+1):
            inds = ~(self.train_ns == n)
            train_xs.append(self.train_xs[inds])
            train_ys.append(self.train_ys[inds])
            train_ts.append(self.train_ts[inds])
            train_ns.append(np.ones(inds.sum(), dtype=int)*n)
            train_subjects.append(self.train_subjects[inds])
        train_xs = np.vstack(train_xs)
        train_ys = np.vstack(train_ys)
        train_ts = np.vstack(train_ts)
        train_ns = np.hstack(train_ns)
        train_subjects = np.hstack(train_subjects)
        full_nonfixations = type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)
        self.full_nonfixations = full_nonfixations
        return full_nonfixations

    def generate_nonfixation_partners(self, seed=42):
        """Generate nonfixational distribution from this
        fixation object such that for every fixation in the
        original fixation object there is a corresponding
        fixation on the same image but on a different
        position that comes from some other fixation.

        This destroys the temporal ordering of the fixation
        trains."""
        train_xs = self.train_xs.copy()
        train_ys = self.train_ys.copy()
        train_ts = self.train_ts.copy()
        train_ns = self.train_ns.copy()
        train_subjects = self.train_subjects.copy()
        rs = np.random.RandomState(seed)
        for train_index in range(len(train_ns)):
            n = train_ns[train_index]
            inds = np.nonzero(self.n != n)[0]
            length = (1 - np.isnan(train_xs[train_index])).sum()
            for i in range(length):
                new_fix_index = rs.choice(inds)
                train_xs[train_index][i] = self.x[new_fix_index]
                train_ys[train_index][i] = self.y[new_fix_index]
                train_ts[train_index][i] = self.t[new_fix_index]
        return type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target):
        """ Write fixationtrains to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.bytes_('FixationTrains')
        target.attrs['version'] = np.bytes_('1.3')

        variable_length_arrays = []

        for attribute in ['train_xs', 'train_ys', 'train_ts', 'train_ns', 'train_subjects', 'train_lengths'] + self.__attributes__:
            if attribute in ['subjects', 'scanpath_index'] + self.auto_attributes:
                continue

            data = getattr(self, attribute)
            if isinstance(data, VariableLengthArray):
                variable_length_arrays.append(attribute)
                data = data._data
            target.create_dataset(attribute, data=data)

        saved_attributes = [attribute_name for attribute_name in self.__attributes__ if attribute_name not in self.auto_attributes]
        target.attrs['__attributes__'] = np.bytes_(json.dumps(saved_attributes))

        target.attrs['scanpath_attribute_mapping'] = np.bytes_(json.dumps(self.scanpath_attribute_mapping))

        scanpath_attributes_group = target.create_group('scanpath_attributes')
        for attribute_name, attribute_value in self.scanpath_attributes.items():
            scanpath_attributes_group.create_dataset(attribute_name, data=attribute_value)
        scanpath_attributes_group.attrs['__attributes__'] = np.bytes_(json.dumps(sorted(self.scanpath_attributes.keys())))

        scanpath_fixation_attributes_group = target.create_group('scanpath_fixation_attributes')
        for attribute_name, attribute_value in self.scanpath_fixation_attributes.items():
            scanpath_fixation_attributes_group.create_dataset(attribute_name, data=attribute_value._data)
        scanpath_fixation_attributes_group.attrs['__attributes__'] = np.bytes_(json.dumps(sorted(self.scanpath_fixation_attributes.keys())))


    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        """ Read train fixations from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'FixationTrains':
            raise ValueError("Invalid type! Expected 'FixationTrains', got", data_type)

        valid_versions = ['1.0', '1.1', '1.2', '1.3']
        if data_version not in valid_versions:
            raise ValueError("Invalid version! Expected one of {}, got {}".format(', '.join(valid_versions), data_version))

        data = {key: source[key][...] for key in ['train_xs', 'train_ys', 'train_ts', 'train_ns', 'train_subjects']}

        json_attributes = decode_string(source.attrs['__attributes__'])

        attribute_names = list(json.loads(json_attributes))

        attributes = {}
        for key in attribute_names:
            if key in ['scanpath_index']:
                continue

            if data_version < '1.3' and key == 'subjects':
                continue

            attributes[key] = source[key][...]

        data['attributes'] = attributes

        if data_version < '1.1':
            data['scanpath_attributes'] = {}
        else:
            data['scanpath_attributes'] = _load_attribute_dict_from_hdf5(source['scanpath_attributes'])

        if data_version < '1.2':
            data['scanpath_fixation_attributes'] = {}
            data['scanpath_attribute_mapping'] = {}
        else:
            data['scanpath_fixation_attributes'] = _load_attribute_dict_from_hdf5(source['scanpath_fixation_attributes'])
            data['scanpath_attribute_mapping'] = json.loads(decode_string(source.attrs['scanpath_attribute_mapping']))

        if data_version < '1.3':
            train_lengths = np.array([len(remove_trailing_nans(data['train_xs'][i])) for i in range(len(data['train_xs']))])
        else:
            train_lengths = source['train_lengths'][...]

        data['scanpath_fixation_attributes'] = {
            key: VariableLengthArray(value, train_lengths) for key, value in data['scanpath_fixation_attributes'].items()
        }

        fixations = cls(**data)

        return fixations


def _scanpath_from_fixation_index(fixations, fixation_index, scanpath_attribute_names, scanpath_fixation_attribute_names):
    history_length = fixations.scanpath_history_length[fixation_index]
    xs = np.hstack((
        fixations.x_hist[fixation_index, :history_length],
        [fixations.x[fixation_index]]
    ))

    ys = np.hstack((
        fixations.y_hist[fixation_index, :history_length],
        [fixations.y[fixation_index]]
    ))

    ts = np.hstack((
        fixations.t_hist[fixation_index, :history_length],
        [fixations.t[fixation_index]]
    ))

    n = fixations.n[fixation_index]

    subject = fixations.subject[fixation_index]

    scanpath_attributes = {
        attribute: getattr(fixations, attribute)[fixation_index]
        for attribute in scanpath_attribute_names
    }

    scanpath_fixation_attributes = {}
    for attribute in scanpath_fixation_attribute_names:
        attribute_value = np.hstack((
            getattr(fixations, '{attribute}_hist'.format(attribute=attribute))[fixation_index, :history_length],
            [getattr(fixations, attribute)[fixation_index]]
        ))
        scanpath_fixation_attributes[attribute] = attribute_value


    return xs, ys, ts, n, subject, scanpath_attributes, scanpath_fixation_attributes


def scanpaths_from_fixations(fixations: Fixations, verbose=False) -> Tuple[ScanpathFixations, np.ndarray]:
    """ reconstructs scanpaths as ScanpathFixations from fixations which originally came from scanpaths.

    when called as in

        scanpath_fixations, indices = scanpaths_from_fixations(fixations)

    you will have scanpath_fixations[indices] == fixations.

    :note
        only works if the original scanpaths only used scanpath_attributes and scanpath_fixation_attribute,
        but not attributes (which should not be used for scanpaths anyway).
    """
    if 'scanpath_index' not in fixations.__attributes__:
        raise NotImplementedError("Fixations with scanpath_index attribute required!")

    scanpath_xs = []
    scanpath_ys = []
    scanpath_ts = []
    scanpath_ns = []
    scanpath_subjects = []
    __attributes__ = [
        attribute for attribute in fixations.__attributes__
        if attribute != 'subject' and attribute != 'scanpath_index' and not attribute.endswith('_hist')
    ]

    __scanpath_attributes__ = [
        attribute for attribute in __attributes__
        if '{attribute}_hist'.format(attribute=attribute) not in fixations.__attributes__
    ]
    __scanpath_fixation_attributes__ = [
        attribute for attribute in __attributes__ if attribute not in __scanpath_attributes__
    ]

    scanpath_fixation_attributes = {attribute: [] for attribute in __scanpath_fixation_attributes__}
    scanpath_attributes = {attribute: [] for attribute in __scanpath_attributes__}

    indices = np.ones(len(fixations), dtype=int) * -1
    fixation_counter = 0

    for scanpath_index in tqdm(sorted(np.unique(fixations.scanpath_index)), disable=not verbose):
        scanpath_indices = fixations.scanpath_index == scanpath_index
        scanpath_integer_indices = np.nonzero(scanpath_indices)[0]
        lengths = fixations.scanpath_history_length[scanpath_indices]

        # build scanpath up to maximum length
        maximum_length = max(lengths)
        _index_of_maximum_length = np.argmax(lengths)
        index_of_maximum_length = scanpath_integer_indices[_index_of_maximum_length]

        xs, ys, ts, n, subject, this_scanpath_attributes, this_scanpath_fixation_attributes = _scanpath_from_fixation_index(
            fixations,
            index_of_maximum_length,
            __scanpath_attributes__,
            __scanpath_fixation_attributes__
        )

        for key, value in this_scanpath_attributes.items():
            other_values = getattr(fixations, key)[scanpath_indices]
            if not np.all(other_values == value):
                raise ValueError("attribute {key} not consistent in scanpath {scanpath_index}, found {other_values}".format(
                    key=key, scanpath_index=scanpath_index, other_values=other_values,
                ))

        scanpath_xs.append(xs)
        scanpath_ys.append(ys)
        scanpath_ts.append(ts)
        scanpath_ns.append(n)
        scanpath_subjects.append(subject)

        for attribute, value in this_scanpath_fixation_attributes.items():
            scanpath_fixation_attributes[attribute].append(value)
        for attribute, value in this_scanpath_attributes.items():
            scanpath_attributes[attribute].append(value)

        # build indices

        for index_in_scanpath in range(maximum_length+1):
            if index_in_scanpath in lengths:
                # add index to indices
                index_in_fixations = scanpath_integer_indices[list(lengths).index(index_in_scanpath)]

                # there might be one fixation multiple times in fixations.
                indices_in_fixations = scanpath_integer_indices[lengths == index_in_scanpath]
                indices[indices_in_fixations] = fixation_counter + index_in_scanpath

        fixation_counter += len(xs)

    scanpath_attributes = {
        attribute: np.array(value) for attribute, value in scanpath_attributes.items()
    }

    attribute_mapping = {}

    scanpaths = Scanpaths(
        xs=scanpath_xs,
        ys=scanpath_ys,
        ts=scanpath_ts,
        n=scanpath_ns,
        subject=scanpath_subjects,
        scanpath_attributes=scanpath_attributes,
        fixation_attributes=scanpath_fixation_attributes,
        attribute_mapping=attribute_mapping
    )

    return ScanpathFixations(scanpaths=scanpaths), indices