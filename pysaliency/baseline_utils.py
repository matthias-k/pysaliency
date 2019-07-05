from __future__ import print_function, unicode_literals, division, absolute_import

import numba
import numpy as np
from scipy.special import logsumexp
from scipy.ndimage.filters import gaussian_filter

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator

from tqdm import tqdm

from .precomputed_models import get_image_hash
from .roc import general_roc
from .numba_utils import fill_fixation_map
from . import Model, UniformModel


@numba.jit(nopython=True)
def _normalize_fixations(orig_xs, orig_ys, orig_ns, sizes, new_xs, new_ys, real_widths, real_heights):
    for i in range(len(orig_xs)):
        height, width = sizes[orig_ns[i]]
        new_xs[i] = orig_xs[i] / width
        new_ys[i] = orig_ys[i] / height
        real_widths[i] = width
        real_heights[i] = height


def normalize_fixations(stimuli, fixations, keep_aspect=False, add_shape=False, verbose=True):
    sizes = np.array(stimuli.sizes)

    xs = np.empty(len(fixations.x))
    ys = np.empty(len(fixations.x))
    widths = np.empty(len(fixations.x))
    heights = np.empty(len(fixations.x))

    _normalize_fixations(fixations.x, fixations.y, fixations.n, sizes,
                         xs, ys, widths, heights)

    real_widths = widths.copy()
    real_heights = heights.copy()

    if keep_aspect:
        max_size = np.max([widths, heights], axis=0)
        widths = max_size
        heights = max_size

    xs = fixations.x / widths
    ys = fixations.y / heights

    real_widths /= widths
    real_heights /= heights

    if add_shape:
        return xs, ys, real_widths, real_heights

    return xs, ys


def fixations_to_scikit_learn(fixations, normalize=None, keep_aspect=False, add_shape=False,
                              add_stimulus_number=False,
                              add_fixation_number=False,
                              verbose=True):
    if normalize is None:
        xs = fixations.x
        ys = fixations.y
        data = [xs, ys]
        if add_shape:
            raise NotImplementedError()
    else:
        data = normalize_fixations(normalize, fixations, keep_aspect=keep_aspect, add_shape=add_shape,
                                   verbose=verbose)
    if add_stimulus_number:
        data = list(data) + [fixations.n]
    if add_fixation_number:
        data = list(data) + [np.arange(len(fixations.n))]
    return np.vstack(data).T.copy()


class ScikitLearnImageCrossValidationGenerator(object):
    def __init__(self, stimuli, fixations):
        self.stimuli = stimuli
        self.fixations = fixations

    def __iter__(self):
        for n in range(len(self.stimuli)):
            inds = self.fixations.n == n
            if inds.sum():
                yield ~inds, inds

    def __len__(self):
        return len(self.stimuli)


class ScikitLearnImageSubjectCrossValidationGenerator(object):
    def __init__(self, stimuli, fixations):
        self.stimuli = stimuli
        self.fixations = fixations

    def __iter__(self):
        for n in range(len(self.stimuli)):
            for s in range(self.fixations.subject_count):
                image_inds = self.fixations.n == n
                subject_inds = self.fixations.subjects == s
                train_inds, test_inds = image_inds & ~subject_inds, image_inds & subject_inds
                if test_inds.sum() == 0 or train_inds.sum() == 0:
                    #print("Skipping")
                    continue
                yield train_inds, test_inds

    def __len__(self):
        return len(set(zip(self.fixations.n, self.fixations.subjects)))


class ScikitLearnWithinImageCrossValidationGenerator(object):
    def __init__(self, stimuli, fixations, chunks_per_image=10, random_seed=42):
        self.stimuli = stimuli
        self.fixations = fixations
        self.chunks_per_image = chunks_per_image
        self.rng = np.random.RandomState(seed=random_seed)

    def __iter__(self):
        for n in range(len(self.stimuli)):
            image_inds = self.fixations.n == n
            _image_inds = np.nonzero(image_inds)[0]
            self.rng.shuffle(_image_inds)
            chunks = np.array_split(_image_inds, self.chunks_per_image)
            for chunk in chunks:
                if not len(chunk):
                    continue
                test_inds = np.zeros_like(self.fixations.n)
                test_inds[chunk] = 1
                test_inds = test_inds > 0.5
                train_inds = image_inds & ~test_inds
                yield train_inds, test_inds

    def __len__(self):
        #counts = 0
        #for n in range(len(self.stimuli)):

        return len(self.stimuli)*self.chunks_per_image


class RegularizedKernelDensityEstimator(BaseEstimator):
    def __init__(self, bandwidth=1.0, regularization = 1.0e-5):
        self.bandwidth = bandwidth
        self.regularization = regularization

    def setup(self):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        height, width = self.shape
        self.uniform_density = -np.log(width*height)

        self.kde_constant = np.log(1-self.regularization)
        self.uniform_constant = np.log(self.regularization)

    def fit(self, X):
        self.shape = X[0, 2:4]
        self.setup()
        self.kde.fit(X[:, 0:2])
        return self

    def score_samples(self, X):
        kde_logliks = self.kde.score_samples(X[:, :2])

        logliks = np.logaddexp(
            self.kde_constant + kde_logliks,
            self.uniform_constant + self.uniform_density
        )
        return logliks

    def score(self, X):
        return np.sum(self.score_samples(X))


class MixtureKernelDensityEstimator(BaseEstimator):
    def __init__(self, bandwidth=1.0, regularization = 1.0e-5, regularizing_log_likelihoods=None):
        self.bandwidth = bandwidth
        self.regularization = regularization
        #self.regularizer_model = regularizer_model
        ##self.stimuli = stimuli
        self.regularizing_log_likelihoods = regularizing_log_likelihoods

    def setup(self):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        self.kde_constant = np.log(1-self.regularization)
        self.uniform_constant = np.log(self.regularization)

    def fit(self, X):
        assert X.shape[1] == 3

        self.setup()
        self.kde.fit(X[:, 0:2])
        return self

    def score_samples(self, X):
        assert X.shape[1] == 3

        kde_logliks = self.kde.score_samples(X[:, :2])
        fix_ns = X[:, 2].astype(int)
        fix_lls = self.regularizing_log_likelihoods[fix_ns]

        logliks = np.logaddexp(
            self.kde_constant + kde_logliks,
            self.uniform_constant + fix_lls
        )
        return logliks

    def score(self, X):
        return np.sum(self.score_samples(X))


class AUCKernelDensityEstimator(BaseEstimator):
    def __init__(self, nonfixations, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.nonfixations = nonfixations

    def setup(self):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

    def fit(self, X):
        self.setup()
        self.kde.fit(X)
        self.nonfixation_values = self.kde.score_samples(self.nonfixations)
        return self

    def score_samples(self, X):
        pos_logliks = self.kde.score_samples(X)
        neg_logliks = self.nonfixation_values

        aucs = [general_roc(np.array([p]), neg_logliks)[0] for p in pos_logliks]

        return aucs

    def score(self, X):
        return np.sum(self.score_samples(X))


class GoldModel(Model):
    def __init__(self, stimuli, fixations, bandwidth, eps = 1e-20, keep_aspect=False, verbose=False, **kwargs):
        super(GoldModel, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.fixations = fixations
        self.bandwidth = bandwidth
        self.eps = eps
        self.keep_aspect = keep_aspect
        self.xs, self.ys = normalize_fixations(stimuli, fixations, keep_aspect=self.keep_aspect, verbose=verbose)
        self.shape_cache = {}

    def _log_density(self, stimulus):
        shape = stimulus.shape[0], stimulus.shape[1]

        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)

        #fixations = self.fixations[self.fixations.n == stimulus_index]
        inds = self.fixations.n == stimulus_index

        if not inds.sum():
            return UniformModel().log_density(stimulus)

        ZZ = np.zeros(shape)
        if self.keep_aspect:
            height, width = shape
            max_size = max(width, height)
            x_factor = max_size
            y_factor = max_size
        else:
            x_factor = shape[1]
            y_factor = shape[0]
        _fixations = np.array([self.ys[inds]*y_factor, self.xs[inds]*x_factor]).T
        fill_fixation_map(ZZ, _fixations)
        ZZ = gaussian_filter(ZZ, [self.bandwidth*y_factor, self.bandwidth*x_factor])
        ZZ *= (1-self.eps)
        ZZ += self.eps * 1.0/(shape[0]*shape[1])
        ZZ = np.log(ZZ)

        ZZ -= logsumexp(ZZ)
        #ZZ -= np.log(np.exp(ZZ).sum())

        return ZZ



class KDEGoldModel(Model):
    def __init__(self, stimuli, fixations, bandwidth, eps = 1e-20, keep_aspect=False, verbose=False, **kwargs):
        super(KDEGoldModel, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.fixations = fixations
        self.bandwidth = bandwidth
        self.eps = eps
        self.keep_aspect = keep_aspect
        self.xs, self.ys = normalize_fixations(stimuli, fixations, keep_aspect=self.keep_aspect, verbose=verbose)
        self.shape_cache = {}

    def _log_density(self, stimulus):
        shape = stimulus.shape[0], stimulus.shape[1]

        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)

        #fixations = self.fixations[self.fixations.n == stimulus_index]
        inds = self.fixations.n == stimulus_index

        if not inds.sum():
            return UniformModel().log_density(stimulus)

        X = fixations_to_scikit_learn(
            self.fixations[inds], normalize=self.stimuli,
            keep_aspect=self.keep_aspect, add_shape=False, verbose=False)
        kde = KernelDensity(bandwidth=self.bandwidth).fit(X)

        height, width = shape
        if self.keep_aspect:
            max_size = max(height, width)
            rel_height = height / max_size
            rel_width = width / max_size
        else:
            rel_height = 1.0
            rel_width = 1.0

        # calculate the KDE score at the middle of each pixel:
        # for a width of 10 pixels, we are going to calculate at
        # 0.5, 1.5, ..., 9.5, since e.g. fixations with x coordinate between 0.0 and 1.0
        # will be evaluated at pixel index 0.
        xs = np.linspace(0, rel_width, num=width, endpoint=False)+0.5*rel_width/width
        ys = np.linspace(0, rel_height, num=height, endpoint=False)+0.5*rel_height/height
        XX, YY = np.meshgrid(xs, ys)
        scores = kde.score_samples(np.column_stack((XX.flatten(), YY.flatten()))).reshape(XX.shape)
        scores -= logsumexp(scores)
        ZZ = scores

        if self.eps:
            ZZ = np.logaddexp(
                np.log(1-self.eps)+scores,
                np.log(self.eps)-np.log(height*width)
            )

        ZZ -= logsumexp(ZZ)

        return ZZ



class CrossvalidatedBaselineModel(Model):
    def __init__(self, stimuli, fixations, bandwidth, eps = 1e-20, **kwargs):
        super(CrossvalidatedBaselineModel, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.fixations = fixations
        self.bandwidth = bandwidth
        self.eps = eps
        self.xs, self.ys = normalize_fixations(stimuli, fixations)
        #self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.vstack([self.xs, self.ys]).T)
        self.shape_cache = {}

    def _log_density(self, stimulus):
        shape = stimulus.shape[0], stimulus.shape[1]

        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)

        #fixations = self.fixations[self.fixations.n == stimulus_index]
        inds = self.fixations.n != stimulus_index

        ZZ = np.zeros(shape)

        _fixations = np.array([self.ys[inds]*shape[0], self.xs[inds]*shape[1]]).T
        fill_fixation_map(ZZ, _fixations)
        ZZ = gaussian_filter(ZZ, [self.bandwidth*shape[0], self.bandwidth*shape[1]])
        ZZ *= (1-self.eps)
        ZZ += self.eps * 1.0/(shape[0]*shape[1])
        ZZ = np.log(ZZ)

        ZZ -= logsumexp(ZZ)
        #ZZ -= np.log(np.exp(ZZ).sum())

        return ZZ


class BaselineModel(Model):
    def __init__(self, stimuli, fixations, bandwidth, eps = 1e-20, keep_aspect=False, **kwargs):
        super(BaselineModel, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.fixations = fixations
        self.bandwidth = bandwidth
        self.eps = eps
        self.keep_aspect = keep_aspect
        self.xs, self.ys = normalize_fixations(stimuli, fixations, keep_aspect=keep_aspect)
        self.shape_cache = {}

    def _log_density(self, stimulus):
        shape = stimulus.shape[0], stimulus.shape[1]
        if shape not in self.shape_cache:
            ZZ = np.zeros(shape)
            height, width = shape
            if self.keep_aspect:
                max_size = max(height, width)
                y_factor = max_size
                x_factor = max_size
            else:
                y_factor = height
                x_factor = width
            _fixations = np.array([self.ys*y_factor, self.xs*x_factor]).T
            fill_fixation_map(ZZ, _fixations)
            ZZ = gaussian_filter(ZZ, [self.bandwidth*y_factor, self.bandwidth*x_factor])
            ZZ *= (1-self.eps)
            ZZ += self.eps * 1.0/(shape[0]*shape[1])
            ZZ = np.log(ZZ)

            ZZ -= logsumexp(ZZ)
            self.shape_cache[shape] = ZZ

        return self.shape_cache[shape]
