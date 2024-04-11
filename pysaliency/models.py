from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod

from itertools import combinations

from boltons.cacheutils import LRU
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from tqdm import tqdm

from .saliency_map_models import (SaliencyMapModel, ScanpathSaliencyMapModel, handle_stimulus,
                                  SubjectDependentSaliencyMapModel,
                                  DensitySaliencyMapModel,
                                  DisjointUnionMixin,
                                  GaussianSaliencyMapModel,
                                  )
from .datasets import Scanpaths, ScanpathFixations, check_prediction_shape, get_image_hash, as_stimulus
from .metrics import probabilistic_image_based_kl_divergence, convert_saliency_map_to_density
from .sampling_models import SamplingModelMixin
from .utils import Cache, average_values, deprecated_class, remove_trailing_nans, iterator_chunks


def _prepare_logprobabilities_for_sampling(log_probabilities):
    log_probabilities = np.asarray(log_probabilities)

    valid_indices = np.nonzero(np.isfinite(log_probabilities))[0]
    valid_log_probabilities = log_probabilities[valid_indices]

    ndxs = valid_log_probabilities.argsort()
    sorted_log_probabilities = valid_log_probabilities[ndxs]
    cumsums = np.logaddexp.accumulate(sorted_log_probabilities)
    cumsums -= cumsums[-1]

    return cumsums, ndxs, valid_indices


def _sample_from_cumsums(cumsums, ndxs, valid_indices, size, rst=None):
    if rst is None:
        rst = np.random

    tmps = -rst.exponential(size=size)
    js = np.searchsorted(cumsums, tmps)
    valid_values = ndxs[js]
    values = valid_indices[valid_values]

    return values


def sample_from_logprobabilities(log_probabilities, size=1, rst=None):
    """ Sample from log probabilities (robust to many bins and small probabilities).

        +-np.inf and np.nan will be interpreted as zero probability
    """

    cumsums, ndxs, valid_indices = _prepare_logprobabilities_for_sampling(log_probabilities)
    values = _sample_from_cumsums(cumsums, ndxs, valid_indices, size, rst=rst)

    return values


def sample_from_logdensity(log_density, count=None, rst=None):
    if count is None:
        real_count = 1
    else:
        real_count = count

    height, width = log_density.shape
    flat_log_density = log_density.flatten(order='C')
    samples = sample_from_logprobabilities(flat_log_density, size=real_count, rst=rst)
    sample_xs = samples % width
    sample_ys = samples // width

    if count is None:
        return sample_xs[0], sample_ys[0]
    else:
        return np.asarray(sample_xs), np.asarray(sample_ys)


class LogDensitySampler(object):
    """use this class if you need to sample repeatedly from the same log density. It will do the
       slow parts (sorting the log density, computing log cum sums etc) only once.
    """
    def __init__(self, log_density):
        self.height, self.width = log_density.shape
        flat_log_density = log_density.flatten(order='C')
        self.cumsums, self.ndxs, self.valid_indices = _prepare_logprobabilities_for_sampling(flat_log_density)

    def sample(self, size, rst=None):
        samples = _sample_from_cumsums(self.cumsums, self.ndxs, self.valid_indices, size, rst=rst)
        sample_xs = samples % self.width
        sample_ys = samples // self.width

        return np.asarray(sample_xs), np.asarray(sample_ys)

    def sample_batch_fixations(self, fixations_per_image, batch_size, rst=None):
        xs, ys = self.sample(fixations_per_image * batch_size, rst=rst)
        ns = np.repeat(np.arange(batch_size, dtype=int), repeats=fixations_per_image)

        return xs, ys, ns


def sample_from_image(densities, count=None, rst=None):
    if rst is None:
        rst = np.random
    height, width = densities.shape
    sorted_densities = densities.flatten(order='C')
    cumsums = np.cumsum(sorted_densities)
    if count is None:
        real_count = 1
    else:
        real_count = count
    sample_xs = []
    sample_ys = []
    tmps = rst.rand(real_count)
    js = np.searchsorted(cumsums, tmps)
    for j in js:
        sample_xs.append(j % width)
        sample_ys.append(j // width)
    sample_xs = np.asarray(sample_xs)
    sample_ys = np.asarray(sample_ys)
    if count is None:
        return sample_xs[0], sample_ys[0]
    else:
        return np.asarray(sample_xs), np.asarray(sample_ys)


class ScanpathModel(SamplingModelMixin, object, metaclass=ABCMeta):
    """
    General probabilistic saliency model.

    Inheriting classes have to implement `conditional_log_density`
    """

    @abstractmethod
    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        raise NotImplementedError()

    def conditional_log_density_for_fixation(self, stimuli, fixations, fixation_index, out=None):
        return self.conditional_log_density(
            stimuli.stimulus_objects[fixations.n[fixation_index]],
            x_hist=remove_trailing_nans(fixations.x_hist[fixation_index]),
            y_hist=remove_trailing_nans(fixations.y_hist[fixation_index]),
            t_hist=remove_trailing_nans(fixations.t_hist[fixation_index]),
            attributes={key: getattr(fixations, key)[fixation_index] for key in fixations.__attributes__},
            out=out
        )

    def conditional_log_densities(self, stimuli, fixations, verbose=False, **kwargs):
        """ returns conditional log density predictions for each fixation """
        return [self.conditional_log_density_for_fixation(stimuli, fixations, fixation_index) for fixation_index in tqdm(range(len(fixations)), disable=not verbose)]

    def log_likelihoods(self, stimuli, fixations, verbose=False):
        log_likelihoods = np.empty(len(fixations.x))
        for i in tqdm(range(len(fixations.x)), disable=not verbose):
            conditional_log_density = self.conditional_log_density_for_fixation(stimuli, fixations, i)
            check_prediction_shape(conditional_log_density, stimuli[fixations.n[i]])
            log_likelihoods[i] = conditional_log_density[fixations.y_int[i], fixations.x_int[i]]

        return log_likelihoods

    def log_likelihood(self, stimuli, fixations, verbose=False, average='fixation'):
        return average_values(self.log_likelihoods(stimuli, fixations, verbose=verbose), fixations, average=average)

    def information_gains(self, stimuli, fixations, baseline_model=None, verbose=False, average='fixation'):
        if average != 'fixation':
            raise NotImplementedError()
        if baseline_model is None:
            baseline_model = UniformModel()

        own_log_likelihoods = self.log_likelihoods(stimuli, fixations, verbose=verbose)
        baseline_log_likelihoods = baseline_model.log_likelihoods(stimuli, fixations, verbose=verbose)
        return (own_log_likelihoods - baseline_log_likelihoods) / np.log(2)

    def information_gain(self, stimuli, fixations, baseline_model=None, verbose=False, average='fixation'):
        return average_values(self.information_gains(stimuli, fixations, baseline_model, verbose=verbose), fixations, average=average)

    def _expand_sample_arguments(self, stimuli, train_counts, lengths=None, stimulus_indices=None):
        if isinstance(train_counts, int):
            train_counts = [train_counts]*len(stimuli)

        if stimulus_indices is None:
            if len(train_counts) > len(stimuli):
                raise ValueError('Number of train counts higher than count of stimuli!')
            stimulus_indices = range(len(train_counts))

        if isinstance(stimulus_indices, int):
            stimulus_indices = [stimulus_indices]

        if len(stimulus_indices) != len(train_counts):
            raise ValueError('Number of train counts must match number of stimulus_indices!')

        if isinstance(lengths, int):
            lengths = [lengths for i in range(len(stimuli))]

        if len(train_counts) != len(lengths):
            raise ValueError('Number of train counts and number of lengths do not match!')

        new_lengths = []
        for k, l in enumerate(lengths):
            if isinstance(l, int):
                ll = [l for i in range(train_counts[k])]
            else:
                ll = l
            new_lengths.append(ll)
        lengths = new_lengths

        for k, (c, ls) in enumerate(zip(train_counts, lengths)):
            if c != len(ls):
                raise ValueError('{}th train count ({}) does not match given number of lengths ({})!'.format(k, c, len(ls)))

        return stimuli, train_counts, lengths, stimulus_indices

    def sample(self, stimuli, train_counts, lengths=1, stimulus_indices=None, rst=None, verbose=False) -> ScanpathFixations:
        """
        Sample fixations for given stimuli


        Examples
        --------

        >>> model.sample(stimuli, 10)  # sample 10 fixations per image
        >>> # sample 5 fixations from the first image, 10 from the second and
        >>> # 30 fixations from the third image
        >>> model.sample(stimuli, [5, 10, 30])
        >>> # Sample 10 fixation trains per image, each consiting of 5 fixations
        >>> model.sample(stimuli, 10, lengths=5)
        >>> # Sample 10 fixation trains per image, consisting of 2 fixations
        >>> # for the first image, 4 fixations for the second, ...
        >>> model.sample(stimuli, 10, lengths=[2, 4, 3])
        >>> # Sample
        >>> model.sample(stimuli, 3, lengths=[[1,2,3], [1,2,3]])

        >>> # Sample 3 fixations from the 10th stimulus
        >>> model.sample(stimuli, 3, stimulus_indices = 10)

        >>> # Sample 3 fixations from the 20th and the 42th stimuli each
        >>> model.sample(stimuli, 3, stimulus_indices = [20, 42])
        """

        stimuli, train_counts, lengths, stimulus_indices = self._expand_sample_arguments(stimuli,
                                                                                         train_counts,
                                                                                         lengths,
                                                                                         stimulus_indices)

        xs = []
        ys = []
        ts = []
        ns = []
        subjects = []
        total_count = sum(len(l) for l in lengths)
        pbar = tqdm(total=total_count, disable=not verbose)
        for stimulus_index, ls in zip(stimulus_indices, lengths):
            stimulus = stimuli[stimulus_index]
            for l in ls:
                this_xs, this_ys, this_ts = self._sample_fixation_train(stimulus, l, rst=rst)
                xs.append(this_xs)
                ys.append(this_ys)
                ts.append(this_ts)
                ns.append(stimulus_index)
                subjects.append(0)
                pbar.update(1)
        return ScanpathFixations(Scanpaths(xs=xs, ys=ys, ts=ts, n=ns, subject=subjects))

    def _sample_fixation_train(self, stimulus, length, rst=None):
        """Sample one fixation train of given length from stimulus"""
        return self.sample_scanpath(stimulus, [], [], [], length, rst=rst)

    def sample_fixation(self, stimulus, x_hist, y_hist, t_hist, attributes=None, verbose=False, rst=None):
        log_densities = self.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes)
        x, y = sample_from_image(np.exp(log_densities), rst=rst)
        return x, y, len(t_hist)


class Model(ScanpathModel):
    """
    Time independend probabilistic saliency model.

    Inheriting classes have to implement `_log_density`.
    """
    def __init__(self, cache_location=None, caching=True, memory_cache_size=None):
        super(Model, self).__init__()
        self._cache = Cache(cache_location, memory_cache_size=memory_cache_size)
        self.caching = caching
        #self._log_density_cache = Cache(cache_location)
        # This make the property `cache_location` work.
        #self._saliency_map_cache = self._log_density_cache

    @property
    def cache_location(self):
        return self._cache.cache_location

    @cache_location.setter
    def cache_location(self, value):
        self._cache.cache_location = value

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        return self.log_density(stimulus)

    def log_density(self, stimulus):
        """
        Get log_density for given stimulus.

        To overwrite this function, overwrite `_log_density` as otherwise
        the caching mechanism is disabled.
        """
        stimulus = handle_stimulus(stimulus)
        if not self.caching:
            return self._log_density(stimulus.stimulus_data)
        stimulus_id = stimulus.stimulus_id
        if not stimulus_id in self._cache:
            self._cache[stimulus_id] = self._log_density(stimulus.stimulus_data)
        return self._cache[stimulus_id]

    @abstractmethod
    def _log_density(self, stimulus):
        """
        Overwrite this to implement you own SaliencyMapModel.

        Parameters
        ----------

        @type  stimulus: ndarray
        @param stimulus: stimulus for which the saliency map should be computed.
        """
        raise NotImplementedError()

    def log_likelihoods(self, stimuli, fixations, verbose=False):
        log_likelihoods = np.empty(len(fixations.x))
        for n in tqdm(range(len(stimuli)), disable=not verbose):
            inds = fixations.n == n
            if not inds.sum():
                continue
            log_density = self.log_density(stimuli[n])
            check_prediction_shape(log_density, stimuli[n])
            this_log_likelihoods = log_density[fixations.y_int[inds], fixations.x_int[inds]]
            log_likelihoods[inds] = this_log_likelihoods

        return log_likelihoods

    def _sample_fixation_train(self, stimulus, length, rst=None):
        """Sample one fixation train of given length from stimulus"""
        # We could reuse the implementation from `ScanpathModel`
        # but this implementation is much faster for long trains.
        log_densities = self.log_density(stimulus)
        xs, ys = sample_from_image(np.exp(log_densities), count=length, rst=rst)
        ts = np.arange(len(xs))
        return xs, ys, ts

    def pixel_space_information_gain(self, baseline, gold_standard, stimulus, eps=1e-20):
        log_p_gold = gold_standard.log_density(stimulus)
        log_p_baseline = baseline.log_density(stimulus)
        log_p_model = self.log_density(stimulus)
        p_gold = np.exp(log_p_gold)
        p_gold[p_gold == 0] = p_gold[p_gold > 0].min()
        ig = (p_gold)*(np.logaddexp(log_p_model, np.log(eps))-np.logaddexp(log_p_baseline, np.log(eps)))
        return ig

    def kl_divergences(self, stimuli, gold_standard, log_regularization=0, quotient_regularization=0, verbose=False):
        """Calculate KL Divergence between model and gold standard for each stimulus.

        This metric works only for probabilistic models.
        For the existing saliency metrics known as KL Divergence, see
        `image_based_kl_divergence` and `fixation_based_kl_divergence`.

        log_regularization and quotient_regularization are regularization constants that are used as in
        kldiv(p1, p2) = sum(p1*log(log_regularization + p1 / (p2 + quotient_regularization))).
        """
        assert isinstance(self, Model)
        assert isinstance(gold_standard, Model)

        kl_divs = []
        for s in tqdm(stimuli, disable=not verbose):
            logp_model = self.log_density(s)
            logp_gold = gold_standard.log_density(s)
            check_prediction_shape(logp_model, s)
            check_prediction_shape(logp_gold, s)
            kl_divs.append(
                probabilistic_image_based_kl_divergence(logp_model, logp_gold, log_regularization=log_regularization, quotient_regularization=quotient_regularization)
            )

        return kl_divs

    def set_params(self, **kwargs):
        """
            Set model parameters, if the model has parameters

            This method has to reset caches etc., if the depend on the parameters
        """
        if kwargs:
            raise ValueError('Unkown parameters!', kwargs)


class CachedModel(Model):
    """Density model which uses only precached densities
    """
    def __init__(self, cache_location, **kwargs):
        if cache_location is None:
            raise ValueError("CachedModel needs a cache location!")
        super(CachedModel, self).__init__(cache_location=cache_location, **kwargs)

    def _log_density(self, stimulus):
        raise NotImplementedError()


class UniformModel(Model):
    """Saliency model assuming uniform fixation distribution over space
    """
    def _log_density(self, stimulus):
        return np.zeros((stimulus.shape[0], stimulus.shape[1])) - np.log(stimulus.shape[0]) - np.log(stimulus.shape[1])

    def log_likelihoods(self, stimuli, fixations, verbose=False):
        stimulus_shapes = np.zeros((len(stimuli), 2), dtype=int)
        stimulus_indices = sorted(np.unique(fixations.n))
        for stimulus_index in stimulus_indices:
            stimulus_shapes[stimulus_index] = stimuli.stimulus_objects[stimulus_index].size

        with np.errstate(divide='ignore'):  # ignore log(0) warnings, we won't use them anyway
            stimulus_log_likelihoods = -np.log(stimulus_shapes).sum(axis=1)

        return stimulus_log_likelihoods[fixations.n]


class MixtureModel(Model):
    """ A saliency model being a weighted mixture of a number of other models
    """
    def __init__(self, models, weights=None, check_norm=True, **kwargs):
        """Create a mixture model from a list of models and a list of weights

           :param models: list of `Model` instances
           :param weights: list of weights for the different models. Do not have
                           to sum up to one, they will be normalized.
                           If `None`, will be set to a uniform mixture.
        """
        super(MixtureModel, self).__init__(**kwargs)
        self.models = models
        if weights is None:
            weights = np.ones(len(self.models))
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        if not len(weights) == len(models):
            raise ValueError('models and weights must have same length!')
        self.weights = weights
        self.check_norm = check_norm

    def _log_density(self, stimulus):
        log_densities = []
        for i, model in enumerate(self.models):
            log_density = model.log_density(stimulus).copy()
            log_density += np.log(self.weights[i])
            log_densities.append(log_density)

        log_density = logsumexp(log_densities, axis=0)

        if self.check_norm:
            np.testing.assert_allclose(np.exp(log_density).sum(), 1.0, rtol=1e-7)

        if not log_density.shape == (stimulus.shape[0], stimulus.shape[1]):
            raise ValueError('wrong density shape in mixture model! stimulus shape: ({}, {}), density shape: {}'.format(stimulus.shape[0], stimulus.shape[1], log_density.shape))
        return log_density


class MixtureScanpathModel(ScanpathModel):
    """ A scanpath model being a weighted mixture of a number of other models
    """
    def __init__(self, models, weights=None, check_norm=True, **kwargs):
        """Create a mixture scanpath model from a list of models and a list of weights

           :param models: list of `ScanpathModel` instances
           :param weights: list of weights for the different models. Do not have
                           to sum up to one, they will be normalized.
                           If `None`, will be set to a uniform mixture.
        """
        super(MixtureScanpathModel, self).__init__(**kwargs)
        self.models = models
        if weights is None:
            weights = np.ones(len(self.models))
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        if not len(weights) == len(models):
            raise ValueError('models and weights must have same length!')
        self.weights = weights
        self.check_norm = check_norm

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        log_densities = []
        for i, model in enumerate(self.models):
            log_density = model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes).copy()
            log_density += np.log(self.weights[i])
            log_densities.append(log_density)

        log_density = logsumexp(log_densities, axis=0)
        if self.check_norm:
            np.testing.assert_allclose(np.exp(log_density).sum(), 1.0, rtol=1e-7)
        if not log_density.shape == (stimulus.shape[0], stimulus.shape[1]):
            raise ValueError('wrong density shape in mixture model! stimulus shape: ({}, {}), density shape: {}'.format(stimulus.shape[0], stimulus.shape[1], log_density.shape))
        return log_density


class ResizingModel(Model):
    def __init__(self, parent_model, verbose=True, **kwargs):
        if 'caching' not in kwargs:
            kwargs['caching'] = False
        self.verbose = verbose
        super(ResizingModel, self).__init__(**kwargs)
        self.parent_model = parent_model

    def _log_density(self, stimulus):
        smap = self.parent_model.log_density(stimulus)

        target_shape = (stimulus.shape[0],
                        stimulus.shape[1])

        if smap.shape != target_shape:
            if self.verbose:
                print("Resizing saliency map", smap.shape, target_shape)
            x_factor = target_shape[1] / smap.shape[1]
            y_factor = target_shape[0] / smap.shape[0]

            smap = zoom(smap, [y_factor, x_factor], order=1, mode='nearest')

            smap -= logsumexp(smap)

            assert smap.shape == target_shape

        return smap


class ResizingScanpathModel(ScanpathModel):
    def __init__(self, parent_model, verbose=True, **kwargs):
        self.verbose = verbose
        super(ResizingScanpathModel, self).__init__(**kwargs)
        self.parent_model = parent_model

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        smap = self.parent_model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes, out=out)

        target_shape = (stimulus.shape[0],
                        stimulus.shape[1])

        if smap.shape != target_shape:
            if self.verbose:
                print("Resizing saliency map", smap.shape, target_shape)
            x_factor = target_shape[1] / smap.shape[1]
            y_factor = target_shape[0] / smap.shape[0]

            smap = zoom(smap, [y_factor, x_factor], order=1, mode='nearest')

            smap -= logsumexp(smap)

            assert smap.shape == target_shape

        return smap


class DisjointUnionModel(DisjointUnionMixin, ScanpathModel):
    def conditional_log_density(self, stimulus, *args, **kwargs):
        raise

    def log_likelihoods(self, stimuli, fixations, **kwargs):
        return self.eval_metric('log_likelihoods', stimuli, fixations, **kwargs)


class SubjectDependentModel(DisjointUnionModel):
    def __init__(self, subject_models, **kwargs):
        super(SubjectDependentModel, self).__init__(**kwargs)
        self.subject_models = subject_models

    def _split_fixations(self, stimuli, fixations):
        for s in self.subject_models:
            yield fixations.subjects == s, self.subject_models[s]

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None, **kwargs):
        if 'subjects' not in attributes:
            raise ValueError("SubjectDependentModel can't compute conditional log densities without subject indication!")
        return self.subject_models[attributes['subjects']].conditional_log_density(
            stimulus, x_hist, y_hist, t_hist, attributes=attributes, **kwargs)

    def get_saliency_map_model_for_sAUC(self, baseline_model):
        return SubjectDependentSaliencyMapModel({
            s: ShuffledAUCSaliencyMapModel(self.subject_models[s], baseline_model)
            for s in self.subject_models})

    def get_saliency_map_model_for_NSS(self):
        return SubjectDependentSaliencyMapModel({
            s: DensitySaliencyMapModel(self.subject_models[s])
            for s in self.subject_models})


class StimulusDependentModel(Model):
    def __init__(self, stimuli_models, check_stimuli=True, fallback_model=None, **kwargs):
        super(StimulusDependentModel, self).__init__(**kwargs)
        self.stimuli_models = stimuli_models
        self.fallback_model = fallback_model
        if check_stimuli:
            self.check_stimuli()

    def check_stimuli(self):
        for s1, s2 in tqdm(list(combinations(self.stimuli_models, 2))):
            if not set(s1.stimulus_ids).isdisjoint(s2.stimulus_ids):
                raise ValueError('Stimuli not disjoint')

    def _log_density(self, stimulus):
        stimulus_hash = get_image_hash(stimulus)
        for stimuli, model in self.stimuli_models.items():
            if stimulus_hash in stimuli.stimulus_ids:
                return model.log_density(stimulus)
        else:
            if self.fallback_model is not None:
                return self.fallback_model.log_density(stimulus)
            else:
                raise ValueError('stimulus not provided by these models')


class StimulusDependentScanpathModel(ScanpathModel):
    def __init__(self, stimuli_models, check_stimuli=True, fallback_model=None, **kwargs):
        super(StimulusDependentScanpathModel, self).__init__(**kwargs)
        self.stimuli_models = stimuli_models
        self.fallback_model = fallback_model
        if check_stimuli:
            self.check_stimuli()

    def check_stimuli(self):
        for s1, s2 in tqdm(list(combinations(self.stimuli_models, 2))):
            if not set(s1.stimulus_ids).isdisjoint(s2.stimulus_ids):
                raise ValueError('Stimuli not disjoint')

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        stimulus_hash = get_image_hash(as_stimulus(stimulus).stimulus_data)
        for stimuli, model in self.stimuli_models.items():
            if stimulus_hash in stimuli.stimulus_ids:
                return model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes, out=out)
        else:
            if self.fallback_model is not None:
                return self.fallback_model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes, out=out)
            else:
                raise ValueError('stimulus not provided by these models')


class FixationIndexDependentModel(ScanpathModel):
    """ a scanpath that uses different models depending of the index of a fixation within a scanpath. """
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.models = models

    def _get_model_for_index(self, fixation_index):
        for (start, end), model in self.models.items():
            if start <= fixation_index < end:
                return model
        raise KeyError(fixation_index)

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        fixation_index = len(remove_trailing_nans(x_hist))
        return self._get_model_for_index(fixation_index).conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes, out=out)


class ShuffledAUCSaliencyMapModel(SaliencyMapModel):
    def __init__(self, probabilistic_model, baseline_model):
        super(ShuffledAUCSaliencyMapModel, self).__init__(caching=False)
        self.probabilistic_model = probabilistic_model
        self.baseline_model = baseline_model

    def _saliency_map(self, stimulus):
        return self.probabilistic_model.log_density(stimulus) - self.baseline_model.log_density(stimulus)


class ShuffledAUCScanpathSaliencyMapModel(ScanpathSaliencyMapModel):
    def __init__(self, probabilistic_model: ScanpathModel, baseline_model: Model):
        super(ShuffledAUCScanpathSaliencyMapModel, self).__init__()
        self.probabilistic_model = probabilistic_model
        self.baseline_model = baseline_model

    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        return (
            self.probabilistic_model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes)
            - self.baseline_model.log_density(stimulus)
        )


def average_predictions(log_densities, log_density_count=None, maximal_chunk_size=None, verbose=False, library='torch'):
    """ compute average log density given multiple log densities.

    specifying log_density_count allows to process generator arrays to avoid keeping all predictions in memory.
    specifying maximal_chunk_size allows to process the log densities such that not too many log densities are kept in
    memory at all (which only makes sense if log_densities is a generator), see logsumexp_iterator for more details
    """

    if maximal_chunk_size is not None and log_density_count is None:
        print("Warning: specifying maximal_chunk_size without log_density_count doesn't make sense because then the log densities have to be converted into a list and hence put in memory anyway.")

    if log_density_count is None:
        log_densities = np.array(list(log_densities))
        log_density_count = len(log_densities)

    normalization_constant = np.log(log_density_count)
    def weighted_log_densities(log_densities, normalization_constant):
        for log_density in log_densities:
            yield log_density - normalization_constant

    result = logsumexp_iterator(weighted_log_densities(log_densities, normalization_constant), log_density_count, maximal_chunk_size=maximal_chunk_size, verbose=verbose, library=library)
    result_norm = logsumexp(result)

    if not (-0.0001 < result_norm < 0.0001):
        print(f"Warning: result of averaging not well normalized (logsum={result_norm}). This could either be a problem with the averaged predictions or indicate numerical issues in averaging.")

    result -= result_norm

    return result


def logsumexp_iterator(log_density_iterator, iterator_length, maximal_chunk_size=10, verbose=False, library='torch'):
    """computes logsumexp of iterator such not too many values are in memory.

    Works by splitting the sequence into shorter chunks, processing them and then adding the results up.
    This is done in a recursive manner: If the chunks are still too long, they are again split up.
    This guarantees that per recursion level never more than `maximal_chunk_size` items are kept in memory.
    """
    if iterator_length is None or maximal_chunk_size is None or (maximal_chunk_size is not None and iterator_length <= maximal_chunk_size):
        if verbose:
            print(f"directly handling {iterator_length} predictions")
        predictions = np.array(list(log_density_iterator))
    else:
        predictions = []
        chunk_size = iterator_length // (maximal_chunk_size // 2)
        if verbose:
            print(f"splitting {iterator_length} predictions up into chunks of length {chunk_size}")
        for i, iterator_chunk in enumerate(iterator_chunks(log_density_iterator, chunk_size=chunk_size)):
            if verbose:
                print(f"handling {i}th chunk of length {chunk_size}")
            predictions.append(logsumexp_iterator(iterator_chunk, chunk_size, maximal_chunk_size=maximal_chunk_size, verbose=verbose))
        predictions = np.array(predictions)
        if verbose:
            print(f"Done handling {iterator_length} predictions in chunks of {chunk_size}")


    if library == 'tensorflow':
        from .tf_utils import tf_logsumexp
        prediction = tf_logsumexp(predictions, axis=0)
    elif library == 'torch':
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prediction = torch.logsumexp(torch.Tensor(predictions).to(device), dim=0).detach().cpu().numpy()
    elif library == 'numpy':
        prediction = logsumexp(predictions, axis=0)
    else:
        raise ValueError(library)

    return prediction


class ShuffledBaselineModel(Model):
    """Predicts a mixture of all predictions for other images.

    This model will usually be used as baseline model for computing sAUC saliency maps.

    To avoid computing many averages over many images, this model will once compute an
    an average over all predictions, and then only remove the prediction for a given
    image when computing model predictions.

    use the library parameter to define whether the logsumexp should be computed
    with torch (default), tensorflow or numpy.

    predict_overall_average: If False (default), for each image, the average over all
    other images in `stimuli` will be predicted. If set to True, simply the average
    over all images in stimuli will be predicted.
    """
    def __init__(self, parent_model, stimuli,
                 compute_size=(500, 500),
                 library='torch',
                 prepopulate_cache=False,
                 maximal_chunk_size=20,
                 predict_overall_average=False,
                 **kwargs):
        super(ShuffledBaselineModel, self).__init__(**kwargs)
        self.parent_model = parent_model
        self.stimuli = stimuli
        self.compute_size = tuple(compute_size)
        self.maximal_chunk_size = maximal_chunk_size
        self.predict_overall_average = predict_overall_average
        self.prediction = None
        if library not in ['torch', 'tensorflow', 'numpy']:
            raise ValueError(library)
        self.library = library

        if prepopulate_cache:
            self.get_average_prediction(verbose=True)

    def get_average_prediction(self, verbose=False):
        if self.prediction is not None:
            return self.prediction


        def log_density_iterable():
            for stimulus in tqdm(self.stimuli, disable=not verbose):
                yield self._resize_prediction(self.parent_model.log_density(stimulus), self.compute_size)

        prediction = average_predictions(
            log_density_iterable(),
            log_density_count=len(self.stimuli),
            maximal_chunk_size=self.maximal_chunk_size,
            library=self.library
        )

        self.prediction = prediction
        return self.prediction

    def _resize_prediction(self, prediction, target_shape):
        if prediction.shape != target_shape:
            orig_shape = prediction.shape
            x_factor = target_shape[1] / prediction.shape[1]
            y_factor = target_shape[0] / prediction.shape[0]

            prediction = zoom(prediction, [y_factor, x_factor], order=1, mode='nearest')

            prediction -= logsumexp(prediction)

            if prediction.shape != target_shape:
                print("compute size", self.compute_size)
                print("prediction shape", orig_shape)
                print("target shape", target_shape)
                print("x factor", x_factor)
                print("y factor", y_factor)
                raise ValueError(prediction.shape)

            assert prediction.shape == target_shape

        return prediction

    def _log_density(self, stimulus):
        average_log_density = self.get_average_prediction()

        if self.predict_overall_average:
            prediction = average_log_density
        else:
            # here we're effectively computing the average prediction of all predictions except for the
            # one for this stimulus, by substracting the current prection from the average prediction
            # with correct weights. This allows us to only once iterate over all predictions at model start.
            this_log_density = self._resize_prediction(self.parent_model.log_density(stimulus), self.compute_size)
            N = len(self.stimuli)
            prediction =np.log(
                np.exp(average_log_density + np.log(N) - np.log(N - 1))
                -np.exp(this_log_density - np.log(N - 1))
            )

        target_shape = (stimulus.shape[0], stimulus.shape[1])
        prediction = self._resize_prediction(prediction, target_shape)

        return prediction

ShuffledSimpleBaselineModel = deprecated_class(deprecated_in='0.2.22', removed_in='1.0.0', details="Use ShuffledBaselineModel instead, which is now as effective as the old ShuffledSimpleBaselineModel")(ShuffledBaselineModel)


class GaussianModel(Model):
    def __init__(self, width=0.5, center_x=0.5, center_y=0.5, **kwargs):
        super(GaussianModel, self).__init__(**kwargs)
        self.parent_model = GaussianSaliencyMapModel(width=width, center_x=center_x, center_y=center_y)

    def _log_density(self, stimulus):
        saliency_map = self.parent_model.saliency_map(stimulus)

        density = saliency_map / saliency_map.sum()
        return np.log(density)


class SaliencyMapNormalizingModel(Model):
    """ Probabilistic model that converts saliency maps into
        fixation densities by dividing by their sum
    """
    def __init__(self, model, minimum_value=0.0):
        self.model = model
        self.minimum_value = minimum_value
        super(SaliencyMapNormalizingModel, self).__init__(caching=False)

    def _log_density(self, stimulus):
        smap = convert_saliency_map_to_density(self.model.saliency_map(stimulus), minimum_value=self.minimum_value)
        return np.log(smap)


class FixedStimulusSizeModel(Model):
    """ model which scales images to have a fixed size before handing them to anothet model"""
    def __init__(self, size, parent_model, verbose=False, **kwargs):
        super(FixedStimulusSizeModel, self).__init__(**kwargs)

        self.size = size
        self.parent_model = parent_model
        self.verbose = verbose

    def _log_density(self, stimulus):
        stimulus = self.ensure_color(stimulus)

        stimulus_size = stimulus.shape[0], stimulus.shape[1]
        max_size = max(stimulus_size)
        factor = self.size / max_size

        if factor != 1.0:
            if self.verbose:
                print("Resizing with factor", factor)
            stimulus_for_parent_model = zoom(stimulus, [factor, factor, 1.0], order=1, mode='nearest')
        else:
            stimulus_for_parent_model = stimulus

        log_density = self.parent_model.log_density(stimulus_for_parent_model)

        factor_y = stimulus.shape[0] / log_density.shape[0]
        factor_x = stimulus.shape[1] / log_density.shape[1]

        if factor_y != 1.0 or factor_x != 1.0:
            if self.verbose:
                print("Wrong shape, resizing log densities", stimulus.shape, log_density.shape)
            log_density = zoom(log_density, [factor_y, factor_x], order=1, mode='nearest')
            log_density -= logsumexp(log_density)

        assert log_density.shape[0] == stimulus.shape[0]
        assert log_density.shape[1] == stimulus.shape[1]

        return log_density

    def ensure_color(self, image):
        if image.ndim == 2:
            return np.dstack((image, image, image))
        return image


class DVAAwareModel(Model):
    """ A model which adapts another model to a new image resolution by rescaling images before computing predictions

    - dva: expected image resolution in pixel per dva for this model
    - parent_model_dva: image resolution expected by parent_model
    """
    def __init__(self, dva, parent_model, parent_model_dva, verbose=False, **kwargs):

        super(DVAAwareModel, self).__init__(**kwargs)

        self.dva = dva
        self.parent_model = parent_model
        self.parent_model_dva = parent_model_dva
        self.verbose = verbose

        self.factor = self.parent_model_dva / self.dva

    def _log_density(self, stimulus):
        stimulus_data = as_stimulus(stimulus).stimulus_data
        stimulus = self.ensure_color(stimulus_data)

        if self.factor != 1.0:
            if self.verbose:
                print("Resizing with factor", self.factor)
            stimulus_for_parent_model = zoom(stimulus, [self.factor, self.factor, 1.0], order=1, mode='nearest')
        else:
            stimulus_for_parent_model = stimulus

        log_density = self.parent_model.log_density(stimulus_for_parent_model)

        factor_y = stimulus.shape[0] / log_density.shape[0]
        factor_x = stimulus.shape[1] / log_density.shape[1]

        if factor_y != 1.0 or factor_x != 1.0:
            if self.verbose:
                print("Wrong shape, resizing log densities", stimulus.shape, log_density.shape)
            log_density = zoom(log_density, [factor_y, factor_x], order=1, mode='nearest')
            log_density -= logsumexp(log_density)

        assert log_density.shape[0] == stimulus.shape[0]
        assert log_density.shape[1] == stimulus.shape[1]

        return log_density

    def ensure_color(self, image):
        if image.ndim == 2:
            return np.dstack((image, image, image))
        return image


class DVAAwareScanpathModel(ScanpathModel):
    """ A scanpath model which adapts another model to a new image resolution by rescaling images before computing predictions

    - dva: expected image resolution in pixel per dva for this model
    - parent_model_dva: image resolution expected by parent_model
    """
    def __init__(self, dva: float, parent_model: ScanpathModel, parent_model_dva: float, verbose=False, **kwargs):

        super(DVAAwareScanpathModel, self).__init__(**kwargs)

        self.dva = dva
        self.parent_model = parent_model
        self.parent_model_dva = parent_model_dva
        self.verbose = verbose

        self.factor = self.parent_model_dva / self.dva

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        stimulus_data = as_stimulus(stimulus).stimulus_data
        stimulus = self.ensure_color(stimulus_data)
        if out is not None:
            raise NotImplementedError()

        if self.factor != 1.0:
            if self.verbose:
                print("Resizing with factor", self.factor)
            stimulus_for_parent_model = zoom(stimulus, [self.factor, self.factor, 1.0], order=1, mode='nearest')

            outer_shape = (
                stimulus.shape[0],
                stimulus.shape[1]
            )

            inner_shape = (
                stimulus_for_parent_model.shape[0],
                stimulus_for_parent_model.shape[1]
            )

            x_factor = outer_shape[1] / inner_shape[1]
            y_factor = outer_shape[0] / inner_shape[0]

            if x_factor != 1:
                x_hist_for_parent_model = np.array(x_hist) / x_factor
            if y_factor != 1:
                y_hist_for_parent_model = np.array(y_hist) / y_factor


        else:
            stimulus_for_parent_model = stimulus
            x_hist_for_parent_model = x_hist
            y_hist_for_parent_model = y_hist


        log_density = self.parent_model.conditional_log_density(
            stimulus=stimulus_for_parent_model,
            x_hist = x_hist_for_parent_model,
            y_hist=y_hist_for_parent_model,
            t_hist=t_hist,
            attributes=attributes
        )

        factor_y = stimulus.shape[0] / log_density.shape[0]
        factor_x = stimulus.shape[1] / log_density.shape[1]

        if factor_y != 1.0 or factor_x != 1.0:
            if self.verbose:
                print("Wrong shape, resizing log densities", stimulus.shape, log_density.shape)
            log_density = zoom(log_density, [factor_y, factor_x], order=1, mode='nearest')
            log_density -= logsumexp(log_density)

        assert log_density.shape[0] == stimulus.shape[0]
        assert log_density.shape[1] == stimulus.shape[1]

        return log_density

    def ensure_color(self, image):
        if image.ndim == 2:
            return np.dstack((image, image, image))
        return image


GeneralModel = deprecated_class(deprecated_in='0.2.16', removed_in='1.0.0', details="Use ScanpathModel instead")(ScanpathModel)
StimulusDependentGeneralModel = deprecated_class(deprecated_in='0.2.16', removed_in='1.0.0', details="Use StimulusDependentScanpathModel instead")(StimulusDependentScanpathModel)
