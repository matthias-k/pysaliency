# vim: set expandtab :
#kate: space-indent on; indent-width 4; backspace-indents on;
from __future__ import absolute_import, print_function, division, unicode_literals

from hashlib import sha1
from copy import deepcopy
from collections import Sequence

from six.moves import range as xrange

import numpy as np
from scipy.misc import imread
from PIL import Image

from .utils import LazyList


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
    def __init__(self, x, y, t, x_hist, y_hist, t_hist, n, subjects):
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
    def __init__(self, train_xs, train_ys, train_ts, train_ns, train_subjects):
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
                self.x_hist[out_index][:fix_index] = self.train_xs[train_index][:fix_index]
                self.y_hist[out_index][:fix_index] = self.train_ys[train_index][:fix_index]
                self.t_hist[out_index][:fix_index] = self.train_ts[train_index][:fix_index]
                out_index += 1
        self.full_nonfixations = None

    def filter_fixation_trains(self, indices):
        """
        Create new fixations object which contains only the fixation trains indicated.
        """
        if self.__attributes__ != ['subjects']:
            raise NotImplementedError('Filtering fixation trains with additional attributes is not yet implemented!')
        train_xs = self.train_xs[indices]
        train_ys = self.train_ys[indices]
        train_ts = self.train_ts[indices]
        train_ns = self.train_ns[indices]
        train_subjects = self.train_subjects[indices]
        return type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)

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
    def from_fixation_trains(cls, xs, ys, ts, ns, subjects):
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
        return cls(train_xs, train_ys, train_ts, train_ns, train_subjects)

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



def get_image_hash(img):
    """
    Calculate a unique hash for the given image.

    Can be used to cache results for images, e.g. saliency maps.
    """
    return sha1(np.ascontiguousarray(img)).hexdigest()


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
    def __init__(self, stimuli):
        self.stimuli = stimuli
        self.shapes = [s.shape for s in self.stimuli]
        self.sizes = LazyList(lambda n: (self.shapes[n][0], self.shapes[n][1]),
                              length = len(self.stimuli))
        self.stimulus_ids = LazyList(lambda n: get_image_hash(self.stimuli[n]),
                                     length=len(self.stimuli),
                                     pickle_cache=True)
        self.stimulus_objects = [StimuliStimulus(self, n) for n in range(len(self.stimuli))]

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ObjectStimuli([self.stimulus_objects[i] for i in range(len(self))[index]])
        elif isinstance(index, list):
            return ObjectStimuli([self.stimulus_objects[i] for i in index])
        else:
            return self.stimulus_objects[index]


class ObjectStimuli(Stimuli):
    """
    This Stimuli class is mainly used for slicing of other stimuli objects.
    """
    def __init__(self, stimulus_objects):
        self.stimulus_objects = stimulus_objects
        self.stimuli = LazyList(lambda n: self.stimulus_objects[n].stimulus_data,
                                length = len(self.stimulus_objects))
        self.shapes = LazyList(lambda n: self.stimulus_objects[n].shape,
                               length = len(self.stimulus_objects))
        self.sizes = LazyList(lambda n: self.stimulus_objects[n].size,
                              length = len(self.stimulus_objects))
        self.stimulus_ids = LazyList(lambda n: self.stimulus_objects[n].stimulus_id,
                                     length = len(self.stimulus_objects))


class FileStimuli(Stimuli):
    """
    Manage a list of stimuli that are saved as files.
    """
    def __init__(self, filenames, cache=True):
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

        self.stimulus_ids = LazyList(lambda n: get_image_hash(self.stimuli[n]),
                                     length=len(self.stimuli),
                                     pickle_cache=True)
        self.stimulus_objects = [StimuliStimulus(self, n) for n in range(len(self.stimuli))]
        self.sizes = LazyList(lambda n: (self.shapes[n][0], self.shapes[n][1]),
                              length = len(self.stimuli))

    def load_stimulus(self, n):
        return imread(self.filenames[n])


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
