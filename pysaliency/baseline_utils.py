from collections import OrderedDict
from typing import List

import numba
import numpy as np
from boltons.iterutils import chunked
from scipy.ndimage.filters import gaussian_filter
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity

from . import Model, UniformModel
from .numba_utils import fill_fixation_map
from .precomputed_models import get_image_hash
from .roc import general_roc
from .utils import inter_and_extrapolate


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


# crossvalidation generators


class ScikitLearnImageCrossValidationGenerator(object):
    def __init__(self, stimuli, fixations, within_stimulus_attributes=None, leave_out_size=1, maximal_source_count=None):
        self.stimuli = stimuli
        self.fixations = fixations
        self.within_stimulus_attributes = within_stimulus_attributes or []
        self.leave_out_size = leave_out_size
        self.maximal_source_count = maximal_source_count
        if self.within_stimulus_attributes and leave_out_size != 1:
            raise NotImplementedError("cannot yet specify both batchsize and within_stimulus_attributes")
        for attribute in self.within_stimulus_attributes:
            if attribute not in self.stimuli.attributes:
                raise ValueError(f"stimulus attribute '{attribute}' not available in given stimuli")

    def __iter__(self):
        if self.leave_out_size == 1:
            elements = chunked(range(len(self.stimuli)), size=1)
        else:
            indices = np.arange(len(self.stimuli))
            np.random.RandomState(seed=42).shuffle(indices)
            elements = chunked(list(indices), size=self.leave_out_size)

        source_selection_rst = np.random.RandomState(seed=23)
        for ns in elements:
            test_inds = np.isin(self.fixations.n, ns)
            train_inds = ~test_inds

            for attribute_name in self.within_stimulus_attributes:
                target_value = self.stimuli.attributes[attribute_name][ns[0]]
                valid_stimulus_indices = np.nonzero(self.stimuli.attributes[attribute_name] == target_value)[0]
                valid_fixation_indices = np.isin(self.fixations.n, valid_stimulus_indices)
                train_inds = train_inds & valid_fixation_indices
            if test_inds.sum():
                if self.maximal_source_count is not None and train_inds.sum() > self.maximal_source_count:
                    train_inds = np.nonzero(train_inds)[0]
                    selected_train_inds = source_selection_rst.choice(
                        train_inds,
                        size=self.maximal_source_count,
                        replace=False
                    )
                    train_inds = np.zeros_like(test_inds, dtype=bool)
                    train_inds[selected_train_inds] = True
                yield train_inds, test_inds

    def __len__(self):
        return int(np.ceil(len(self.stimuli) / self.leave_out_size))


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


# Scikit-learn compatible estimators for baseline models


class GeneralMixtureKernelDensityEstimator(DensityMixin, BaseEstimator):
    """
    computes the log likelihood of data under a mixture of a kernel density estimator and multiple
    other regularizations.

    Other regulariations are given by their log likelihoods for each sample, where X must contain
    sample indices in the last column. Previous columns will be used for the KDE.

    bandwidth: bandwidth of the kernel density estimator
    regularizations: list of regularization weights of the regularizations. The sum of the weights must be <= 1.0,
            the difference to 1 will the the weight of the KDE.
    regularizing_log_likelihoods: list of log likelihoods of the regularizations for samples. The second dimension
            must match the length of regularizations. The first dimension will be indexed by the last dimension
            of the handed over samples.
    """
    def __init__(self, bandwidth: float, regularizations: List[float], regularizing_log_likelihoods: List[float]):
        self.bandwidth = bandwidth
        self.regularizations = np.asarray(regularizations)
        self.regularizing_log_likelihoods = np.asarray(regularizing_log_likelihoods)

        if not len(self.regularizations) == self.regularizing_log_likelihoods.shape[1]:
            raise ValueError("regularizations and regularizing_log_likelihoods don't match")

    def setup(self):
        assert np.sum(self.regularizations) <= 1.0
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        self.kde_constant = np.log(1-self.regularizations.sum())
        self.regularization_constants = np.log(self.regularizations)

    def fit(self, X):
        assert X.shape[1] == 3

        self.setup()
        self.kde.fit(X[:, 0:2])
        return self

    def score_samples(self, X):
        assert X.shape[1] == 3

        kde_logliks = self.kde.score_samples(X[:, :2])
        fix_inds = X[:, 2].astype(int)
        fix_lls = self.regularizing_log_likelihoods[fix_inds]

        logliks = logsumexp(np.hstack([(self.kde_constant + kde_logliks)[:, np.newaxis],
                                       self.regularization_constants + fix_lls
                                       ]), axis=-1)

        return logliks

    def score(self, X):
        return np.sum(self.score_samples(X))


class RegularizedKernelDensityEstimator(DensityMixin, BaseEstimator):
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


class MixtureKernelDensityEstimator(DensityMixin, BaseEstimator):
    def __init__(self, bandwidth=1.0, regularization = 1.0e-5, regularizing_log_likelihoods=None):
        self.bandwidth = bandwidth
        self.regularization = regularization
        self.regularizing_log_likelihoods = np.asarray(regularizing_log_likelihoods)

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


class AUCKernelDensityEstimator(DensityMixin, BaseEstimator):
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


# Classes for computing crossvalidation scores of fixations on stimuli on KDE models
# with multiple regularization models

def _normalize_regularization_factors(args):
    """ makes sure that sum(10**args) <= 1.0, i.e. they can be used as regularizing weights """
    log_regularizations = np.asarray(args)
    for i, value in enumerate(log_regularizations):
        if value >= 0:
            log_regularizations[i] = -1e-10

    for i in list(range(len(log_regularizations)))[::-1]:
        if np.sum([10**value for value in log_regularizations]) <= 1.0:
            break
        #else:
        #    print("not normal", np.sum([10**value for value in log_regularizations]))
        new_value = 1.0 - (10**log_regularizations).sum()
        if new_value < 0:
            new_value = -10
        else:
            new_value = np.log10(new_value)
        log_regularizations[i] = new_value

    return log_regularizations


class CrossvalMultipleRegularizations(object):
    """Class for computing crossvalidation scores of a fixation KDE with multiple regularization models

    n_jobs: number of parallel jobs to use in cross_val_score
    verbose: verbosity level for cross_val_score
    """
    def __init__(self, stimuli, fixations, regularization_models: OrderedDict, crossvalidation, n_jobs=None, verbose=False):
        self.stimuli = stimuli
        self.fixations = fixations

        self.cv = crossvalidation
        self.n_jobs = n_jobs
        self.verbose = verbose

        X_areas = fixations_to_scikit_learn(
            self.fixations, normalize=stimuli,
            keep_aspect=True,
            add_shape=True,
            verbose=False
        )


        self.X = fixations_to_scikit_learn(
            self.fixations,
            normalize=self.stimuli,
            keep_aspect=True, add_shape=False, add_fixation_number=True, verbose=False
        )

        stimuli_sizes = np.array(self.stimuli.sizes)
        real_areas = stimuli_sizes[self.fixations.n, 0] * stimuli_sizes[self.fixations.n, 1]
        areas_gold = X_areas[:, 2] * X_areas[:, 3]
        self.mean_area = np.mean(areas_gold)

        correction = np.log(areas_gold) - np.log(real_areas)
        self.regularization_log_likelihoods = []

        self.regularization_models = []
        self.params = ['log_bandwidth']
        for model_name, model in regularization_models.items():
            model_lls = model.log_likelihoods(self.stimuli, self.fixations, verbose=True)
            self.regularization_log_likelihoods.append(model_lls - correction)
            self.params.append('log_{}'.format(model_name))

        self.regularization_log_likelihoods = np.asarray(self.regularization_log_likelihoods).T

    def score(self, log_bandwidth, *args, **kwargs):
        for i, arg in enumerate(args):
            name = self.params[i+1]
            if name in kwargs:
                raise ValueError("double arguments!", args, kwargs)
            kwargs[name] = arg
        log_regularizations = np.array([kwargs[k] for k in self.params[1:]])
        log_regularizations = _normalize_regularization_factors(log_regularizations)

        val = cross_val_score(GeneralMixtureKernelDensityEstimator(
            bandwidth=10**log_bandwidth,
            regularizations=10**log_regularizations,
            regularizing_log_likelihoods=self.regularization_log_likelihoods),
            self.X, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs).sum() / len(self.X) / np.log(2)
        val += np.log2(self.mean_area)
        return val


class CrossvalGoldMultipleRegularizations(CrossvalMultipleRegularizations):
    def __init__(self, stimuli, fixations, regularization_models, n_jobs=None, verbose=False):
        if fixations.subject_count > 1:
            crossvalidation_factory = ScikitLearnImageSubjectCrossValidationGenerator
        else:
            crossvalidation_factory = ScikitLearnWithinImageCrossValidationGenerator

        super().__init__(stimuli, fixations, regularization_models, crossvalidation_factory=crossvalidation_factory, n_jobs=n_jobs, verbose=verbose)


# baseline models

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
    def __init__(self, stimuli, fixations, bandwidth, eps=1e-20, keep_aspect=False, verbose=False, grid_spacing=1, **kwargs):
        super(KDEGoldModel, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.bandwidth = bandwidth
        self.eps = eps
        self.keep_aspect = keep_aspect
        self.grid_spacing = grid_spacing
        self.X = fixations_to_scikit_learn(
            fixations, normalize=self.stimuli,
            keep_aspect=self.keep_aspect, add_shape=False, verbose=False)
        self.stimulus_indices = fixations.n
        self.shape_cache = {}

    def _log_density(self, stimulus):
        shape = stimulus.shape[0], stimulus.shape[1]

        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)

        inds = self.stimulus_indices == stimulus_index

        if not inds.sum():
            return UniformModel().log_density(stimulus)

        X = self.X[inds]
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
        xs = np.linspace(0, rel_width, num=width, endpoint=False) + 0.5 * rel_width / width
        ys = np.linspace(0, rel_height, num=height, endpoint=False) + 0.5 * rel_height / height

        if self.grid_spacing > 1:
            xs = xs[::self.grid_spacing]
            ys = ys[::self.grid_spacing]

        XX, YY = np.meshgrid(xs, ys)
        XX_flat = XX.flatten()
        YY_flat = YY.flatten()

        scores = kde.score_samples(np.column_stack((XX_flat, YY_flat)))

        if self.grid_spacing == 1:
            scores = scores.reshape((height, width))
        else:
            x_coordinates = np.arange(0, width)[::self.grid_spacing]
            y_coordinates = np.arange(0, height)[::self.grid_spacing]
            XX_coordinates, YY_coordinates = np.meshgrid(x_coordinates, y_coordinates)
            score_grid = np.empty((height, width)) * np.nan
            score_grid[YY_coordinates.flatten(), XX_coordinates.flatten()] = scores
            score_grid = inter_and_extrapolate(score_grid)
            scores = score_grid

        scores -= logsumexp(scores)
        ZZ = scores

        if self.eps:
            ZZ = np.logaddexp(
                np.log(1 - self.eps) + scores,
                np.log(self.eps) - np.log(height * width)
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
