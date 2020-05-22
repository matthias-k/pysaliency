# vim: set expandtab :
#kate: space-indent on; indent-width 4; backspace-indents on;
from __future__ import absolute_import, print_function, division, unicode_literals

import os
from hashlib import sha1
from collections import Sequence
import json
from functools import wraps
from weakref import WeakValueDictionary

from six.moves import range as xrange
from six import string_types

from boltons.cacheutils import cached
import numpy as np
from imageio import imread
from PIL import Image
from tqdm import tqdm

from .utils import LazyList, build_padded_2d_array


def hdf5_wrapper(mode=None):
    def decorator(f):
        @wraps(f)
        def wrapped(self, target, *args, **kwargs):
            if isinstance(target, str):
                import h5py
                with h5py.File(target, mode) as hdf5_file:
                    return f(self, hdf5_file, *args, **kwargs)
            else:
                return f(self, target, *args, **kwargs)

        return wrapped
    return decorator


def decode_string(data):
    if not isinstance(data, string_types):
        return data.decode('utf8')

    return data


def _split_crossval(fixations, part, partcount):
    xs = []
    ys = []
    ts = []
    ns = []
    subjects = []
    N = int((fixations.train_ns.max()+1)/partcount)
    for n in range(N):
        this_inds = np.nonzero(fixations.train_ns == partcount*n+part)[0]
        new_inds = this_inds
        for i in new_inds:
            xs.append(fixations.train_xs[i])
            ys.append(fixations.train_ys[i])
            ts.append(fixations.train_ts[i])
            ns.append(n)
            subjects.append(fixations.train_subjects[i])
    new_fixations = Fixations.from_fixation_trains(xs, ys, ts, ns, subjects)
    return new_fixations


def read_hdf5(source):
    if isinstance(source, str):
        return _read_hdf5_from_file(source)

    data_type = decode_string(source.attrs['type'])

    if data_type == 'Fixations':
        return Fixations.read_hdf5(source)
    elif data_type == 'FixationTrains':
        return FixationTrains.read_hdf5(source)
    elif data_type == 'Stimuli':
        return Stimuli.read_hdf5(source)
    elif data_type == 'FileStimuli':
        return FileStimuli.read_hdf5(source)
    else:
        raise ValueError("Invalid HDF content type:", data_type)


@cached(WeakValueDictionary())
def _read_hdf5_from_file(source):
    import h5py
    with h5py.File(source, 'r') as hdf5_file:
        return read_hdf5(hdf5_file)


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
    __attributes__ = ['subjects']

    def __init__(self, x, y, t, x_hist, y_hist, t_hist, n, subjects, attributes=None):
        x = np.asarray(x)
        y = np.asarray(y)
        t = np.asarray(t)
        n = np.asarray(n)
        self.x = x
        self.y = y
        self.t = t
        self.x_hist = x_hist
        self.y_hist = y_hist
        self.t_hist = t_hist
        self.n = n
        self.subjects = subjects
        self.lengths = (1 - np.isnan(self.x_hist)).sum(axis=-1)

        if attributes is not None:
            self.__attributes__ = list(self.__attributes__)
            for name, value in attributes.items():
                setattr(self, name, value)

    @classmethod
    def create_without_history(cls, x, y, n, subjects=None):
        """ Create new fixation object from fixation data without time and optionally
            without subject information
        """
        N = len(x)
        t = np.zeros(N)
        x_hist = np.empty((N, 1))*np.nan
        y_hist = np.empty((N, 1))*np.nan
        t_hist = np.empty((N, 1))*np.nan
        if subjects is None:
            subjects = np.ones(N)
        return cls(x, y,  t, x_hist, y_hist, t_hist, n, subjects)

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

    def __getitem__(self, indices):
        return self.filter(indices)

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
            if name != 'subjects':
                other_attributes[name] = kwargs.pop(name)

        new_fix = Fixations(**kwargs)
        for key, value in other_attributes.items():
            setattr(new_fix, key, value)
        new_fix.__attributes__ = list(self.__attributes__)
        return new_fix

    def _get_previous_values(self, name, index):
        """return fixations.name[np.arange(len(fixations.name)),index]"""
        a = getattr(self, name)
        inds = np.arange(len(a))
        if index >= 0:
            return a[inds, index]
        else:
            indexes = self.lengths + index
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
        return np.asarray(self.x, np.int)

    @property
    def y_int(self):
        """ y coordinates of the fixations, converted to integers """
        return np.asarray(self.y, np.int)

    @property
    def subject_count(self):
        return self.subjects.max()+1

    def copy(self):
        cfix = Fixations(self.x.copy(), self.y.copy(), self.t.copy(),
                         self.x_hist.copy(), self.y_hist.copy(), self.t_hist.copy(),
                         self.n.copy(), self.subjects.copy())
        cfix.__attributes__ = list(self.__attributes__)
        for name in self.__attributes__:
            setattr(cfix, name, getattr(self, name).copy())
        return cfix

    @classmethod
    def FixationsWithoutHistory(cls, x, y, t, n, subjects):
        x_hist = np.empty((len(x), 1))*np.nan
        y_hist = np.empty((len(x), 1))*np.nan
        t_hist = np.empty((len(x), 1))*np.nan
        return cls(x, y, t, x_hist, y_hist, t_hist, n, subjects)

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target):
        """ Write fixations to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.string_('Fixations')
        target.attrs['version'] = np.string_('1.0')

        for attribute in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n'] + self.__attributes__:
            target.create_dataset(attribute, data=getattr(self, attribute))

        #target.create_dataset('__attributes__', data=self.__attributes__)
        target.attrs['__attributes__'] = np.string_(json.dumps(self.__attributes__))

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        """ Read fixations from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'Fixations':
            raise ValueError("Invalid type! Expected 'Fixations', got", data_type)

        if data_version != '1.0':
            raise ValueError("Invalid version! Expected '1.0', got", data_version)

        data = {key: source[key][...] for key in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n', 'subjects']}
        fixations = cls(**data)

        json_attributes = source.attrs['__attributes__']
        if not isinstance(json_attributes, string_types):
            json_attributes = json_attributes.decode('utf8')
        __attributes__ = json.loads(json_attributes)
        fixations.__attributes__ == list(__attributes__)

        for key in __attributes__:
            setattr(fixations, key, source[key][...])

        return fixations


class FixationTrains(Fixations):
    """
    Capsules the fixations of a dataset as fixation trains.

    Additionally to `Fixations`, `FixationTrains`-instances
    have the attributes
        train_xs: 2d array (number_of_trains, maximum_length_of_train)
        train_ys: 2d array (number_of_trains, maximum_length_of_train)
        train_ts: 2d array (number_of_trains, maximum_length_of_train)
        train_ns: 1d array (number_of_trains)
        train_subjects: 1d array (number_of_trains)

    """
    def __init__(self, train_xs, train_ys, train_ts, train_ns, train_subjects, scanpath_attributes=None, attributes=None):
        self.__attributes__ = list(self.__attributes__)
        self.__attributes__.append('scanpath_index')
        self.train_xs = train_xs
        self.train_ys = train_ys
        self.train_ts = train_ts
        self.train_ns = train_ns
        self.train_subjects = train_subjects
        N_trains = self.train_xs.shape[0] * self.train_xs.shape[1] - np.isnan(self.train_xs).sum()
        max_length_trains = self.train_xs.shape[1]

        # Create conditional fixations
        self.x = np.empty(N_trains)
        self.y = np.empty(N_trains)
        self.t = np.empty(N_trains)
        self.x_hist = np.empty((N_trains, max_length_trains - 1))
        self.y_hist = np.empty((N_trains, max_length_trains - 1))
        self.t_hist = np.empty((N_trains, max_length_trains - 1))
        self.x_hist[:] = np.nan
        self.y_hist[:] = np.nan
        self.t_hist[:] = np.nan
        self.n = np.empty(N_trains, dtype=int)
        self.lengths = np.empty(N_trains, dtype=int)
        self.subjects = np.empty(N_trains, dtype=int)
        self.scanpath_index = np.empty(N_trains, dtype=int)
        out_index = 0
        for train_index in range(self.train_xs.shape[0]):
            fix_length = (1 - np.isnan(self.train_xs[train_index])).sum()
            for fix_index in range(fix_length):
                self.x[out_index] = self.train_xs[train_index][fix_index]
                self.y[out_index] = self.train_ys[train_index][fix_index]
                self.t[out_index] = self.train_ts[train_index][fix_index]
                self.n[out_index] = self.train_ns[train_index]
                self.subjects[out_index] = self.train_subjects[train_index]
                self.lengths[out_index] = fix_index
                self.scanpath_index[out_index] = train_index
                self.x_hist[out_index][:fix_index] = self.train_xs[train_index][:fix_index]
                self.y_hist[out_index][:fix_index] = self.train_ys[train_index][:fix_index]
                self.t_hist[out_index][:fix_index] = self.train_ts[train_index][:fix_index]
                out_index += 1

        if scanpath_attributes is not None:
            assert isinstance(scanpath_attributes, dict)
            self.scanpath_attributes = scanpath_attributes
        else:
            self.scanpath_attributes = {}

        if attributes:
            self.__attributes__ = list(self.__attributes__)
            for key, value in attributes.items():
                assert key != 'subjects'
                assert key != 'scanpath_index'
                assert len(value) == len(self.x)
                self.__attributes__.append(key)
                value = np.array(value)
                setattr(self, key, value)

        self.full_nonfixations = None

    def filter_fixation_trains(self, indices):
        """
        Create new fixations object which contains only the fixation trains indicated.
        """
        if not set(self.__attributes__).issubset(['subjects', 'scanpath_index']):
            raise NotImplementedError('Filtering fixation trains with additional attributes is not yet implemented!')
        train_xs = self.train_xs[indices]
        train_ys = self.train_ys[indices]
        train_ts = self.train_ts[indices]
        train_ns = self.train_ns[indices]
        train_subjects = self.train_subjects[indices]
        scanpath_attributes = {key: value[inds] for key, value in self.scanpath_attributes.items()}
        return type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects, scanpath_attributes=scanpath_attributes)

    def fixation_trains(self):
        """Yield for every fixation train of the dataset:
             xs, ys, ts, n, subject
        """
        for i in xrange(self.train_xs.shape[0]):
            length = (1 - np.isnan(self.train_xs[i])).sum()
            xs = self.train_xs[i][:length]
            ys = self.train_ys[i][:length]
            ts = self.train_ts[i][:length]
            n = self.train_ns[i]
            subject = self.train_subjects[i]
            yield xs, ys, ts, n, subject

    @classmethod
    def from_fixation_trains(cls, xs, ys, ts, ns, subjects, attributes=None, scanpath_attributes=None):
        """ Create Fixation object from fixation trains.
              - xs, ys, ts: Lists of array_like of double. Each array has to contain
                    the data from one fixation train.
              - ns, subjects: lists of int. ns has to contain the image index for
                    each fixation train, subjects the subject index for each
                    fixation train
        """
        maxlength = max([len(x_train) for x_train in xs])
        train_xs = np.empty((len(xs), maxlength))
        train_xs[:] = np.nan
        train_ys = np.empty((len(xs), maxlength))
        train_ys[:] = np.nan
        train_ts = np.empty((len(xs), maxlength))
        train_ts[:] = np.nan
        train_ns = np.empty(len(xs), dtype=int)
        train_subjects = np.empty(len(xs), dtype=int)
        for i in range(len(train_xs)):
            length = len(xs[i])
            train_xs[i, :length] = xs[i]
            train_ys[i, :length] = ys[i]
            train_ts[i, :length] = ts[i]
            #print ns[i], train_ns[i]
            train_ns[i] = ns[i]
            train_subjects[i] = subjects[i]
        return cls(train_xs, train_ys, train_ts, train_ns, train_subjects, attributes=attributes, scanpath_attributes=scanpath_attributes)

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

#    def generate_nonfixations(self, seed=42):
#        """Generate nonfixational distribution from this
#        fixation object by shuffling the images of the
#        fixation trains. The individual fixation trains
#        will be left intact"""
#        train_xs = self.train_xs.copy()
#        train_ys = self.train_ys.copy()
#        train_ts = self.train_ts.copy()
#        train_ns = self.train_ns.copy()
#        train_subjects = self.train_subjects.copy()
#        max_n = train_ns.max()
#        rs = np.random.RandomState(seed)
#        for i in range(len(train_ns)):
#            old_n = train_ns[i]
#            new_ns = range(0, old_n)+range(old_n+1, max_n+1)
#            new_n = rs.choice(new_ns)
#            train_ns[i] = new_n
#        return type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)
#
#    def generate_more_nonfixations(self, count=1, seed=42):
#        """Generate nonfixational distribution from this
#        fixation object by assining each fixation
#        train to $count other images.
#
#        with count=0, each train will be assigned to all
#        other images"""
#        train_xs = []
#        train_ys = []
#        train_ts = []
#        train_ns = []
#        train_subjects = []
#        max_n = self.train_ns.max()
#        if count == 0:
#            count = max_n-1
#        rs = np.random.RandomState(seed)
#        for i in range(len(self.train_ns)):
#            old_n = self.train_ns[i]
#            new_ns = range(0, old_n)+range(old_n+1, max_n+1)
#            new_ns = rs.choice(new_ns, size=count, replace=False)
#            for new_n in new_ns:
#                train_xs.append(self.train_xs[i])
#                train_ys.append(self.train_ys[i])
#                train_ts.append(self.train_ts[i])
#                train_ns.append(new_n)
#                train_subjects.append(self.train_subjects[i])
#        train_xs = np.vstack(train_xs)
#        train_ys = np.vstack(train_ys)
#        train_ts = np.vstack(train_ts)
#        train_ns = np.hstack(train_ns)
#        train_subjects = np.hstack(train_subjects)
#        # reorder
#        inds = np.argsort(train_ns)
#        train_xs = train_xs[inds]
#        train_ys = train_ys[inds]
#        train_ts = train_ts[inds]
#        train_ns = train_ns[inds]
#        train_subjects = train_subjects[inds]
#        return type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)

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
            train_ns.append(np.ones(inds.sum(), dtype=np.int)*n)
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
            train_ns.append(np.ones(inds.sum(), dtype=np.int)*n)
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
        """ Write fixations to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.string_('FixationTrains')
        target.attrs['version'] = np.string_('1.1')

        for attribute in ['train_xs', 'train_ys', 'train_ts', 'train_ns', 'train_subjects'] + self.__attributes__:
            if attribute in ['subjects', 'scanpath_index']:
                continue
            target.create_dataset(attribute, data=getattr(self, attribute))

        target.attrs['__attributes__'] = np.string_(json.dumps(self.__attributes__))

        scanpath_attributes_group = target.create_group('scanpath_attributes')
        for attribute_name, attribute_value in self.scanpath_attributes.items():
            scanpath_attributes_group.create_dataset(attribute_name, data=attribute_value)
        scanpath_attributes_group.attrs['__attributes__'] = np.string_(json.dumps(sorted(self.scanpath_attributes.keys())))

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        """ Read train fixations from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'FixationTrains':
            raise ValueError("Invalid type! Expected 'FixationTrains', got", data_type)

        valid_versions = ['1.0', '1.1']
        if data_version not in valid_versions:
            raise ValueError("Invalid version! Expected one of {}, got {}".format(', '.join(valid_versions), data_version))

        data = {key: source[key][...] for key in ['train_xs', 'train_ys', 'train_ts', 'train_ns', 'train_subjects']}

        json_attributes = decode_string(source.attrs['__attributes__'])

        attribute_names = list(json.loads(json_attributes))

        attributes = {}
        for key in attribute_names:
            if key in ['subjects', 'scanpath_index']:
                continue

            attributes[key] = source[key][...]

        data['attributes'] = attributes

        if data_version < '1.1':
            scanpath_attributes = {}
        else:
            scanpath_attributes_group = source['scanpath_attributes']

            json_attributes = scanpath_attributes_group.attrs['__attributes__']
            if not isinstance(json_attributes, string_types):
                json_attributes = json_attributes.decode('utf8')
            __attributes__ = json.loads(json_attributes)

            scanpath_attributes = {attribute: scanpath_attributes_group[attribute][...] for attribute in __attributes__}

        data['scanpath_attributes'] = scanpath_attributes

        fixations = cls(**data)

        return fixations


def get_image_hash(img):
    """
    Calculate a unique hash for the given image.

    Can be used to cache results for images, e.g. saliency maps.
    """
    if isinstance(img, Stimulus):
        return img.stimulus_id
    return sha1(np.ascontiguousarray(img)).hexdigest()


def as_stimulus(img_or_stimulus):
    if isinstance(img_or_stimulus, Stimulus):
        return img_or_stimulus

    return Stimulus(img_or_stimulus)


class Stimulus(object):
    """
    Manages a stimulus.

    In application, this can always be substituted by
    the numpy array containing the actual stimulus. This class
    is just there to be able to cache calculation results and
    retrieve the cache content without having to load
    the actual stimulus
    """
    def __init__(self, stimulus_data, stimulus_id = None, shape = None, size = None):
        self.stimulus_data = stimulus_data
        self._stimulus_id = stimulus_id
        self._shape = shape
        self._size = size

    @property
    def stimulus_id(self):
        if self._stimulus_id is None:
            self._stimulus_id = get_image_hash(self.stimulus_data)
        return self._stimulus_id

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.stimulus_data.shape
        return self._shape

    @property
    def size(self):
        if self._size is None:
            self._size = self.stimulus_data.shape[0], self.stimulus_data.shape[1]
        return self._size


class StimuliStimulus(Stimulus):
    """
    Stimulus bound to a Stimuli object
    """
    def __init__(self, stimuli, index):
        self.stimuli = stimuli
        self.index = index

    @property
    def stimulus_data(self):
        return self.stimuli.stimuli[self.index]

    @property
    def stimulus_id(self):
        return self.stimuli.stimulus_ids[self.index]

    @property
    def shape(self):
        return self.stimuli.shapes[self.index]

    @property
    def size(self):
        return self.stimuli.sizes[self.index]


class Stimuli(Sequence):
    """
    Manages a list of stimuli (i.e. images).

    The stimuli can be given as numpy arrays. Using the class `FileStimuli`, the stimuli
    can also be saved on disk and will be loaded only when needed.

    Attributes
    ----------
    stimuli :
        The stimuli as list of numpy arrays
    shapes :
        The shapes of the stimuli. For a grayscale stimulus this will
        be a 2-tuple, for a color stimulus a 3-tuple
    sizes :
        The sizes of all stimuli in pixels as pairs (height, width). In difference
        to `shapes`, the color dimension is ignored here.
    stimulus_ids:
        A unique id for each stimulus. Can be used to cache results for stimuli
    stimulus_objects:
        A `Stimulus` instance for each stimulus. Mainly for caching.

    """
    __attributes__ = []
    def __init__(self, stimuli, attributes=None):
        self.stimuli = stimuli
        self.shapes = [s.shape for s in self.stimuli]
        self.sizes = LazyList(lambda n: (self.shapes[n][0], self.shapes[n][1]),
                              length = len(self.stimuli))
        self.stimulus_ids = LazyList(lambda n: get_image_hash(self.stimuli[n]),
                                     length=len(self.stimuli),
                                     pickle_cache=True)
        self.stimulus_objects = [StimuliStimulus(self, n) for n in range(len(self.stimuli))]

        if attributes is not None:
            assert isinstance(attributes, dict)
            self.attributes = attributes
            self.__attributes__ = list(attributes.keys())
        else:
            self.attributes = {}

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, index):
        if isinstance(index, slice):
            attributes = {key: value[index] for key, value in self.attributes.items()}
            return ObjectStimuli([self.stimulus_objects[i] for i in range(len(self))[index]], attributes=attributes)
        elif isinstance(index, list):
            attributes = {key: value[index] for key, value in self.attributes.items()}
            return ObjectStimuli([self.stimulus_objects[i] for i in index], attributes=attributes)
        else:
            return self.stimulus_objects[index]

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target, verbose=False, compression='gzip', compression_opts=9):
        """ Write stimuli to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.string_('Stimuli')
        target.attrs['version'] = np.string_('1.1')

        for n, stimulus in enumerate(tqdm(self.stimuli, disable=not verbose)):
            target.create_dataset(str(n), data=stimulus, compression=compression, compression_opts=compression_opts)

        for attribute_name, attribute_value in self.attributes.items():
            target.create_dataset(attribute_name, data=attribute_value)
        target.attrs['__attributes__'] = np.string_(json.dumps(self.__attributes__))

        target.attrs['size'] = len(self)

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        """ Read stimuli from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'Stimuli':
            raise ValueError("Invalid type! Expected 'Stimuli', got", data_type)

        if data_version not in ['1.0', '1.1']:
            raise ValueError("Invalid version! Expected '1.0' or '1.1', got", data_version)

        size = source.attrs['size']
        stimuli = []

        for n in range(size):
            stimuli.append(source[str(n)][...])

        if data_version < '1.1':
            __attributes__ = []
        else:
            json_attributes = source.attrs['__attributes__']
            if not isinstance(json_attributes, string_types):
                json_attributes = json_attributes.decode('utf8')
            __attributes__ = json.loads(json_attributes)

        attributes = {attribute: source[attribute][...] for attribute in __attributes__}

        stimuli = cls(stimuli=stimuli, attributes=attributes)


        return stimuli


class ObjectStimuli(Stimuli):
    """
    This Stimuli class is mainly used for slicing of other stimuli objects.
    """
    def __init__(self, stimulus_objects, attributes=None):
        self.stimulus_objects = stimulus_objects
        self.stimuli = LazyList(lambda n: self.stimulus_objects[n].stimulus_data,
                                length = len(self.stimulus_objects))
        self.shapes = LazyList(lambda n: self.stimulus_objects[n].shape,
                               length = len(self.stimulus_objects))
        self.sizes = LazyList(lambda n: self.stimulus_objects[n].size,
                              length = len(self.stimulus_objects))
        self.stimulus_ids = LazyList(lambda n: self.stimulus_objects[n].stimulus_id,
                                     length = len(self.stimulus_objects))

        if attributes is not None:
            assert isinstance(attributes, dict)
            self.attributes = attributes
            self.__attributes__ = list(attributes.keys())
        else:
            self.attributes = {}


    def read_hdf5(self, target):
        raise NotImplementedError()


class FileStimuli(Stimuli):
    """
    Manage a list of stimuli that are saved as files.
    """
    def __init__(self, filenames, cache=True, shapes=None, attributes=None):
        """
        Create a stimuli object that reads it's stimuli from files.

        The stimuli are loaded lazy: each stimulus will be opened not
        before it is accessed. At creation time, all files are opened
        to read their dimensions, however the actual image data won't
        be read.

        .. note ::

            To calculate the stimulus_ids, the stimuli have to be
            loaded. Therefore it might be a good idea to load all
            stimuli and pickle the `FileStimuli` afterwarts. Then
            the ids are pickled but the stimuli will be reloaded
            when needed again.

        Parameters
        ----------
        filenames : list of strings
            filenames of the stimuli
        cache : bool, defaults to True
            whether loaded stimuli should be cached. The cache is excluded from pickling.
        """
        self.filenames = filenames
        self.stimuli = LazyList(self.load_stimulus, len(self.filenames), cache=cache)
        if shapes is None:
            self.shapes = []
            for f in filenames:
                img = Image.open(f)
                size = img.size
                if len(img.mode) > 1:
                    # PIL uses (width, height), we use (height, width)
                    self.shapes.append((size[1], size[0], len(img.mode)))
                else:
                    self.shapes.append((size[1], size[0]))
                del img
        else:
            self.shapes = shapes

        self.stimulus_ids = LazyList(lambda n: get_image_hash(self.stimuli[n]),
                                     length=len(self.stimuli),
                                     pickle_cache=True)
        self.stimulus_objects = [StimuliStimulus(self, n) for n in range(len(self.stimuli))]
        self.sizes = LazyList(lambda n: (self.shapes[n][0], self.shapes[n][1]),
                              length = len(self.stimuli))

        if attributes is not None:
            assert isinstance(attributes, dict)
            self.attributes = attributes
            self.__attributes__ = list(attributes.keys())
        else:
            self.attributes = {}

    def load_stimulus(self, n):
        return imread(self.filenames[n])

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = list(range(len(self)))[index]

        if isinstance(index, list):
            filenames = [self.filenames[i] for i in index]
            shapes = [self.shapes[i] for i in index]
            attributes = {key: [value[i] for i in index] for key, value in self.attributes.items()}
            return type(self)(filenames=filenames, shapes=shapes, attributes=attributes)
        else:
            return self.stimulus_objects[index]

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target):
        """ Write FileStimuli to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.string_('FileStimuli')
        target.attrs['version'] = np.string_('2.1')

        import h5py
        # make sure everything is unicode

        hdf5_filename = target.file.filename
        hdf5_directory = os.path.dirname(hdf5_filename)

        relative_filenames = [os.path.relpath(filename, hdf5_directory) for filename in self.filenames]
        decoded_filenames = [decode_string(filename) for filename in relative_filenames]
        encoded_filenames = [filename.encode('utf8') for filename in decoded_filenames]

        target.create_dataset(
            'filenames',
            data=np.array(encoded_filenames),
            dtype=h5py.special_dtype(vlen=str)
        )

        shape_dataset = target.create_dataset(
            'shapes',
            (len(self), ),
            dtype=h5py.special_dtype(vlen=np.dtype('int64'))
        )

        for n, shape in enumerate(self.shapes):
            shape_dataset[n] = np.array(shape)

        for attribute_name, attribute_value in self.attributes.items():
            target.create_dataset(attribute_name, data=attribute_value)
        target.attrs['__attributes__'] = np.string_(json.dumps(self.__attributes__))

        target.attrs['size'] = len(self)

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source, cache=True):
        """ Read FileStimuli from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'FileStimuli':
            raise ValueError("Invalid type! Expected 'Stimuli', got", data_type)

        valid_versions = ['1.0', '2.0', '2.1']
        if data_version not in valid_versions:
            raise ValueError("Invalid version! Expected one of {}, got {}".format(', '.join(valid_versions), data_version))

        encoded_filenames = source['filenames'][...]

        filenames = [decode_string(filename) for filename in encoded_filenames]

        if data_version >= '2.0':
            hdf5_filename = source.file.filename
            hdf5_directory = os.path.dirname(hdf5_filename)
            filenames = [os.path.join(hdf5_directory, filename) for filename in filenames]

        shapes = [list(shape) for shape in source['shapes'][...]]

        if data_version < '2.1':
            __attributes__ = []
        else:
            json_attributes = source.attrs['__attributes__']
            if not isinstance(json_attributes, string_types):
                json_attributes = json_attributes.decode('utf8')
            __attributes__ = json.loads(json_attributes)

        attributes = {attribute: source[attribute][...] for attribute in __attributes__}

        stimuli = cls(filenames=filenames, cache=cache, shapes=shapes, attributes=attributes)

        return stimuli


def create_subset(stimuli, fixations, stimuli_indices):
    """Create subset of stimuli and fixatins using only stimuli
    with given indices.
    """
    new_stimuli = stimuli[stimuli_indices]
    fix_inds = np.in1d(fixations.n, stimuli_indices)
    new_fixations = fixations[fix_inds]

    index_list = list(stimuli_indices)
    new_pos = {i: index_list.index(i) for i in index_list}
    new_fixation_ns = [new_pos[i] for i in new_fixations.n]
    new_fixations.n = np.array(new_fixation_ns)

    return new_stimuli, new_fixations


def concatenate_stimuli(stimuli):
    attributes = {}
    for key in stimuli[0].attributes.keys():
        attributes[key] = concatenate_attributes(s.attributes[key] for s in stimuli)
    return ObjectStimuli(sum([s.stimulus_objects for s in stimuli], []), attributes=attributes)


def concatenate_attributes(attributes):
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

#np.testing.assert_allclose(concatenate_attributes([[0], [1, 2, 3]]), [0,1,2,3])
#np.testing.assert_allclose(concatenate_attributes([[[0]], [[1],[2], [3]]]), [[0],[1],[2],[3]])
#np.testing.assert_allclose(concatenate_attributes([[[0.,1.]], [[1.],[2.], [3.]]]), [[0, 1],[1,np.nan],[2,np.nan],[3,np.nan]])


def concatenate_fixations(fixations):
    kwargs = {}
    for key in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n', 'subjects']:
        kwargs[key] = concatenate_attributes(getattr(f, key) for f in fixations)
    new_fixations = Fixations(**kwargs)
    attributes = set(fixations[0].__attributes__)
    for f in fixations:
        attributes = attributes.intersection(f.__attributes__)
    attributes = sorted(attributes, key=fixations[0].__attributes__.index)
    for key in attributes:
        if key == 'subjects':
            continue
        setattr(new_fixations, key, concatenate_attributes(getattr(f, key) for f in fixations))

    new_fixations.__attributes__ = attributes

    return new_fixations


def concatenate_datasets(stimuli, fixations):
    """Concatenate multiple Stimuli instances with associated fixations"""

    stimuli = list(stimuli)
    fixations = list(fixations)
    assert len(stimuli) == len(fixations)
    if len(stimuli) == 1:
        return stimuli[0], fixations[0]

    for i in range(len(fixations)):
        offset = sum(len(s) for s in stimuli[:i])
        f = fixations[i].copy()
        f.n += offset
        fixations[i] = f

    return concatenate_stimuli(stimuli), concatenate_fixations(fixations)


def remove_out_of_stimulus_fixations(stimuli, fixations):
    """ Return all fixations which do not occour outside the stimulus
    """
    widths = np.array([s[1] for s in stimuli.sizes])
    heights = np.array([s[0] for s in stimuli.sizes])

    inds = ((fixations.x >= 0) & (fixations.y >= 0) &
            (fixations.x < widths[fixations.n]) &
            (fixations.y < heights[fixations.n])
            )
    return fixations[inds]


def calculate_nonfixation_factors(stimuli, index):
    widths = np.asarray([s[1] for s in stimuli.sizes]).astype(float)
    heights = np.asarray([s[0] for s in stimuli.sizes]).astype(float)

    x_factors = stimuli.sizes[index][1] / widths
    y_factors = stimuli.sizes[index][0] / heights

    return x_factors, y_factors


def create_nonfixations(stimuli, fixations, index, adjust_n = True, adjust_history=True):
    """Create nonfixations from fixations for given index

    stimuli of different sizes will be rescaled to match the
    target stimulus
    """

    x_factors, y_factors = calculate_nonfixation_factors(stimuli, index)

    non_fixations = fixations[fixations.n != index]
    other_ns = non_fixations.n

    non_fixations.x = non_fixations.x * x_factors[other_ns]
    non_fixations.y = non_fixations.y * y_factors[other_ns]

    if adjust_history:
        non_fixations.x_hist = non_fixations.x_hist * x_factors[other_ns][:, np.newaxis]
        non_fixations.y_hist = non_fixations.y_hist * y_factors[other_ns][:, np.newaxis]

    if adjust_n:
        non_fixations.n = np.ones(len(non_fixations.n), dtype=int)*index

    return non_fixations
