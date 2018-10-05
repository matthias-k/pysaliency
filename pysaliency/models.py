from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from six import add_metaclass

from itertools import combinations

import numpy as np
from scipy.ndimage import zoom
from scipy.misc import logsumexp
from tqdm import tqdm

from .generics import progressinfo
from .saliency_map_models import (GeneralSaliencyMapModel, SaliencyMapModel, handle_stimulus,
                                  SubjectDependentSaliencyMapModel,
                                  ExpSaliencyMapModel, DisjointUnionSaliencyMapModel)
from .datasets import FixationTrains, get_image_hash, as_stimulus
from .utils import Cache


def sample_from_logprobabilities(log_probabilities, size=1, rst=None):
    """ Sample from log probabilities (robust to many bins and small probabilities).

        +-np.inf and np.nan will be interpreted as zero probability
    """
    if rst is None:
        rst = np.random
    log_probabilities = np.asarray(log_probabilities)

    valid_indices = np.nonzero(np.isfinite(log_probabilities))[0]
    valid_log_probabilities = log_probabilities[valid_indices]

    ndxs = valid_log_probabilities.argsort()
    sorted_log_probabilities = valid_log_probabilities[ndxs]
    cumsums = np.logaddexp.accumulate(sorted_log_probabilities)
    cumsums -= cumsums[-1]

    tmps = -rst.exponential(size=size)
    js = np.searchsorted(cumsums, tmps)
    valid_values = ndxs[js]
    values = valid_indices[valid_values]

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


@add_metaclass(ABCMeta)
class GeneralModel(object):
    """
    General probabilistic saliency model.

    Inheriting classes have to implement `conditional_log_density`
    """

    @abstractmethod
    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, out=None):
        raise NotImplementedError()

    def log_likelihoods(self, stimuli, fixations, verbose=False):
        log_likelihoods = np.empty(len(fixations.x))
        for i in progressinfo(range(len(fixations.x)), verbose=verbose):
            conditional_log_density = self.conditional_log_density(stimuli.stimulus_objects[fixations.n[i]],
                                                                   fixations.x_hist[i],
                                                                   fixations.y_hist[i],
                                                                   fixations.t_hist[i])
            log_likelihoods[i] = conditional_log_density[fixations.y_int[i], fixations.x_int[i]]

        return log_likelihoods

    def log_likelihood(self, stimuli, fixations, verbose=False):
        return np.mean(self.log_likelihoods(stimuli, fixations, verbose=verbose))

    def information_gains(self, stimuli, fixations, baseline_model=None, verbose=False):
        if baseline_model is None:
            baseline_model = UniformModel()

        own_log_likelihoods = self.log_likelihoods(stimuli, fixations, verbose=verbose)
        baseline_log_likelihoods = baseline_model.log_likelihoods(stimuli, fixations, verbose=verbose)
        return (own_log_likelihoods - baseline_log_likelihoods) / np.log(2)

    def information_gain(self, stimuli, fixations, baseline_model=None, verbose=False):
        return np.mean(self.information_gains(stimuli, fixations, baseline_model, verbose=verbose))

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

    def sample(self, stimuli, train_counts, lengths=1, stimulus_indices=None):
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
        for stimulus_index, ls in progressinfo(zip(stimulus_indices, lengths)):
            stimulus = stimuli[stimulus_index]
            for l in ls:
                this_xs, this_ys, this_ts = self._sample_fixation_train(stimulus, l)
                xs.append(this_xs)
                ys.append(this_ys)
                ts.append(this_ts)
                ns.append(stimulus_index)
                subjects.append(0)
        return FixationTrains.from_fixation_trains(xs, ys, ts, ns, subjects)

    def _sample_fixation_train(self, stimulus, length):
        """Sample one fixation train of given length from stimulus"""
        xs = []
        ys = []
        ts = []
        for i in range(length):
            log_densities = self.conditional_log_density(stimulus, xs, ys, ts)
            x, y = sample_from_image(np.exp(log_densities))
            xs.append(x)
            ys.append(y)
            ts.append(len(ts))

        return xs, ys, ts


class Model(GeneralModel):
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

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, out=None):
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

    def saliency_map(self, stimulus):
        return self.log_density(stimulus)

    def _saliency_map(self, stimulus):
        # We have to implement this abstract method
        pass

    def log_likelihoods(self, stimuli, fixations, verbose=False):
        log_likelihoods = np.empty(len(fixations.x))
        for n in tqdm(range(len(stimuli)), disable=not verbose):
            inds = fixations.n == n
            if not inds.sum():
                continue
            log_density = self.log_density(stimuli.stimulus_objects[n])
            this_log_likelihoods = log_density[fixations.y_int[inds], fixations.x_int[inds]]
            log_likelihoods[inds] = this_log_likelihoods

        return log_likelihoods

    def _sample_fixation_train(self, stimulus, length):
        """Sample one fixation train of given length from stimulus"""
        # We could reuse the implementation from `GeneralModel`
        # but this implementation is much faster for long trains.
        log_densities = self.log_density(stimulus)
        xs, ys = sample_from_image(np.exp(log_densities), count=length)
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

    def kl_divergences(self, stimuli, gold_standard):
        """Calculate KL Divergence between model and gold standard for each stimulus.

        This metric works only for probabilistic models.
        For the existing saliency metrics known as KL Divergence, see
        `image_based_kl_divergence` and `fixation_based_kl_divergence`.
        """
        assert isinstance(self, Model)
        assert isinstance(gold_standard, Model)
        kl_divs = []
        for s in stimuli:
            logp_model = self.log_density(s)
            logp_gold = gold_standard.log_density(s)
            kl_divs.append((np.exp(logp_gold)*(logp_gold - logp_model)).sum())
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
        lls = []
        for n in fixations.n:
            lls.append(-np.log(stimuli.shapes[n][0]) - np.log(stimuli.shapes[n][1]))
        return np.array(lls)


class MixtureModel(Model):
    """ A saliency model being a weighted mixture of a number of other models
    """
    def __init__(self, models, weights=None, **kwargs):
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

    def _log_density(self, stimulus):
        log_densities = []
        for i, model in enumerate(self.models):
            log_density = model.log_density(stimulus).copy()
            log_density += np.log(self.weights[i])
            log_densities.append(log_density)

        log_density = logsumexp(log_densities, axis=0)
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


class DisjointUnionModel(GeneralModel, DisjointUnionSaliencyMapModel):
    def conditional_log_density(self, stimulus, *args, **kwargs):
        raise

    def log_likelihoods(self, stimuli, fixations, **kwargs):
        return self.eval_metric('log_likelihoods', stimuli, fixations, **kwargs)


class SubjectDependentModel(DisjointUnionModel, SubjectDependentSaliencyMapModel):
    def get_saliency_map_model_for_sAUC(self, baseline_model):
        return SubjectDependentSaliencyMapModel({
            s: ShuffledAUCSaliencyMapModel(self.subject_models[s], baseline_model)
            for s in self.subject_models})

    def get_saliency_map_model_for_NSS(self):
        return SubjectDependentSaliencyMapModel({
            s: ExpSaliencyMapModel(self.subject_models[s])
            for s in self.subject_models})


class StimulusDependentModel(Model):
    def __init__(self, stimuli_models, check_stimuli=True, **kwargs):
        super(StimulusDependentModel, self).__init__(**kwargs)
        self.stimuli_models = stimuli_models
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
            raise ValueError('stimulus not provided by these models')


class StimulusDependentGeneralModel(GeneralModel):
    def __init__(self, stimuli_models, check_stimuli=True, **kwargs):
        super(StimulusDependentGeneralModel, self).__init__(**kwargs)
        self.stimuli_models = stimuli_models
        if check_stimuli:
            self.check_stimuli()

    def check_stimuli(self):
        for s1, s2 in tqdm(list(combinations(self.stimuli_models, 2))):
            if not set(s1.stimulus_ids).isdisjoint(s2.stimulus_ids):
                raise ValueError('Stimuli not disjoint')

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, out=None):
        stimulus_hash = get_image_hash(as_stimulus(stimulus).stimulus_data)
        for stimuli, model in self.stimuli_models.items():
            if stimulus_hash in stimuli.stimulus_ids:
                return model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, out=out)
        else:
            raise ValueError('stimulus not provided by these models')


class ShuffledAUCSaliencyMapModel(SaliencyMapModel):
    def __init__(self, probabilistic_model, baseline_model):
        super(ShuffledAUCSaliencyMapModel, self).__init__(caching=False)
        self.probabilistic_model = probabilistic_model
        self.baseline_model = baseline_model

    def _saliency_map(self, stimulus):
        return self.probabilistic_model.log_density(stimulus) - self.baseline_model.log_density(stimulus)
