# vim: set expandtab :
#kate: space-indent on; indent-width 4; backspace-indents on;
from __future__ import absolute_import, print_function, division

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

    def filter(self, inds):
        """create new fixations object which contains only the fixations with indexes in inds"""
        new_fixations = type(self)(self.train_xs, self.train_ys, self.train_ts, self.train_ns, self.train_subjects)

        def filter_array(name):
            a = getattr(self, name).copy()[inds]
            setattr(new_fixations, name, a)
        for name in ['x', 'y', 't', 'x_hist', 'y_hist', 't_hist', 'n', 'lengths', 'subjects']:
            filter_array(name)
        return new_fixations

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

    def generate_nonfixations(self, seed=42):
        """Generate nonfixational distribution from this
        fixation object by shuffling the images of the
        fixation trains. The individual fixation trains
        will be left intact"""
        train_xs = self.train_xs.copy()
        train_ys = self.train_ys.copy()
        train_ts = self.train_ts.copy()
        train_ns = self.train_ns.copy()
        train_subjects = self.train_subjects.copy()
        max_n = train_ns.max()
        rs = np.random.RandomState(seed)
        for i in range(len(train_ns)):
            old_n = train_ns[i]
            new_ns = range(0, old_n)+range(old_n+1, max_n+1)
            new_n = rs.choice(new_ns)
            train_ns[i] = new_n
        return type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)

    def generate_more_nonfixations(self, count=1, seed=42):
        """Generate nonfixational distribution from this
        fixation object by assining each fixation
        train to $count other images.

        with count=0, each train will be assigned to all
        other images"""
        train_xs = []
        train_ys = []
        train_ts = []
        train_ns = []
        train_subjects = []
        max_n = self.train_ns.max()
        if count == 0:
            count = max_n-1
        rs = np.random.RandomState(seed)
        for i in range(len(self.train_ns)):
            old_n = self.train_ns[i]
            new_ns = range(0, old_n)+range(old_n+1, max_n+1)
            new_ns = rs.choice(new_ns, size=count, replace=False)
            for new_n in new_ns:
                train_xs.append(self.train_xs[i])
                train_ys.append(self.train_ys[i])
                train_ts.append(self.train_ts[i])
                train_ns.append(new_n)
                train_subjects.append(self.train_subjects[i])
        train_xs = np.vstack(train_xs)
        train_ys = np.vstack(train_ys)
        train_ts = np.vstack(train_ts)
        train_ns = np.hstack(train_ns)
        train_subjects = np.hstack(train_subjects)
        # reorder
        inds = np.argsort(train_ns)
        train_xs = train_xs[inds]
        train_ys = train_ys[inds]
        train_ts = train_ts[inds]
        train_ns = train_ns[inds]
        train_subjects = train_subjects[inds]
        return type(self)(train_xs, train_ys, train_ts, train_ns, train_subjects)

    def generate_full_nonfixations(self):
        """Generate nonfixational distribution from this
        fixation object by using all fixation trains of
        other images. The individual fixation trains
        will be left intact"""
        if self.full_nonfixations is not None:
            print("Reusing nonfixations!")
            return self.full_nonfixations
        train_xs = []
        train_ys = []
        train_ts = []
        train_ns = []
        train_subjects = []
        #max_n = train_ns.max()
#        new_train_
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


class Stimuli(object):
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
    """
    def __init__(self, stimuli):
        self.stimuli = stimuli
        self.shapes = [s.shape for s in self.stimuli]

    @property
    def sizes(self):
        return [(s[0], s[1]) for s in self.shapes]


class FileStimuli(Stimuli):
    """
    Manage a list of stimuli that are saved as files. The
    """
    def __init__(self, filenames, cache=True):
        """
        Create a stimuli object that reads it's stimuli from files.

        The stimuli are loaded lazy: each stimulus will be opened not
        before it is accessed. At creation time, all files are opened
        to read their dimensions, however the actual image data won't
        be read.


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
            self.shapes.append(img.shape)
            del img

    def load_stimulus(self, n):
        return imread(self.filenames[n])

