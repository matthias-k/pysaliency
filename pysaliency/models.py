from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod
import sys

import numpy as np
import theano
import theano.tensor as T

from optpy import minimize

import generics
from .saliency_map_models import GeneralSaliencyMapModel, SaliencyMapModel, handle_stimulus
from .datasets import FixationTrains
from .utils import Cache
from .theano_utils import SaliencyMapProcessing


def sample_from_image(densities, count=None):
    height, width = densities.shape
    sorted_densities = densities.flatten(order='C')
    cumsums = np.cumsum(sorted_densities)
    if count is None:
        real_count = 1
    else:
        real_count = count
    sample_xs = []
    sample_ys = []
    tmps = np.random.rand(real_count)
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


class GeneralModel(GeneralSaliencyMapModel):
    """
    General probabilistic saliency model.

    Inheriting classes have to implement `conditional_log_density`
    """

    @abstractmethod
    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, out=None):
        raise NotImplementedError()

    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist, out=None):
        return self.conditional_log_density(stimulus, x_hist, y_hist, t_hist, out=out)

    def log_likelihoods(self, stimuli, fixations):
        log_likelihoods = np.empty(len(fixations.x))
        for i in generics.progressinfo(range(len(fixations.x))):
            conditional_log_density = self.conditional_log_density(stimuli.stimulus_objects[fixations.n[i]],
                                                                   fixations.x_hist[i],
                                                                   fixations.y_hist[i],
                                                                   fixations.t_hist[i])
            log_likelihoods[i] = conditional_log_density[fixations.y_int[i], fixations.x_int[i]]

        return log_likelihoods

    def log_likelihood(self, stimuli, fixations):
        return np.mean(self.log_likelihoods(stimuli, fixations))

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
        for stimulus_index, ls in generics.progressinfo(zip(stimulus_indices, lengths)):
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


class Model(GeneralModel, SaliencyMapModel):
    """
    Time independend probabilistic saliency model.

    Inheriting classes have to implement `_log_density`.
    """
    def __init__(self, cache_location=None):
        super(Model, self).__init__()
        self._log_density_cache = Cache(cache_location)
        # This make the property `cache_location` work.
        self._saliency_map_cache = self._log_density_cache

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, out=None):
        return self.log_density(stimulus)

    def log_density(self, stimulus):
        """
        Get log_density for given stimulus.

        To overwrite this function, overwrite `_log_density` as otherwise
        the caching mechanism is disabled.
        """
        stimulus = handle_stimulus(stimulus)
        stimulus_id = stimulus.stimulus_id
        if not stimulus_id in self._log_density_cache:
            self._log_density_cache[stimulus_id] = self._log_density(stimulus.stimulus_data)
        return self._log_density_cache[stimulus_id]

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

    def _sample_fixation_train(self, stimulus, length):
        """Sample one fixation train of given length from stimulus"""
        # We could reuse the implementation from `GeneralModel`
        # but this implementation is much faster for long trains.
        log_densities = self.log_density(stimulus)
        xs, ys = sample_from_image(np.exp(log_densities), count=length)
        ts = np.arange(len(xs))
        return xs, ys, ts


class SaliencyMapConvertor(Model):
    """
    Convert saliency map models to probabilistic models.
    """
    def __init__(self, saliency_map_model, nonlinearity = None, centerbias = None, alpha=1.0, blur_radius = 0,
                 saliency_min = None, saliency_max = None, cache_location = None):
        """
        Parameters
        ----------

        @type  saliency_map_model : SaliencyMapModel
        @param saliency_map_model : The saliency map model to convert to a probabilistic model

        @type  nonlinearity : ndarray
        @param nonlinearity : The nonlinearity to apply. By default the identity

        TODO

        @type  saliency_min, saliency_max: float
        @param saliency_min, saliency_max: The saliency values that are interpreted as 0 and 1 before applying
                                           the nonlinearity. If `None`, the minimum rsp. maximum of each saliency
                                           map will be used.
        """
        super(SaliencyMapConvertor, self).__init__(cache_location=cache_location)
        self.saliency_map_model = saliency_map_model
        if nonlinearity is None:
            nonlinearity = np.linspace(0, 1.0, num=20)
        if centerbias is None:
            centerbias = np.ones(12)

        self._blur_radius = blur_radius
        self._nonlinearity = nonlinearity
        self._centerbias = centerbias
        self._alpha = alpha

        self.saliency_min = saliency_min
        self.saliency_max = saliency_max

        self._build()

    def _build(self):
        self.theano_input = T.matrix('saliency_map', dtype='float64')
        self.saliency_map_processing = SaliencyMapProcessing(self.theano_input,
                                                             sigma=self._blur_radius,
                                                             nonlinearity_ys=self._nonlinearity,
                                                             centerbias=self._centerbias,
                                                             alpha=self._alpha
                                                             )
        self._f_log_density = theano.function([self.theano_input], self.saliency_map_processing.log_density)

    def _prepare_saliency_map(self, saliency_map):
        smin, smax = self.saliency_min, self.saliency_max
        if smin is None:
            smin = saliency_map.min()
        if smax is None:
            smax = saliency_map.max()

        saliency_map = (saliency_map - smin) / (smax - smin)
        return saliency_map

    def _log_density(self, stimulus):
        saliency_map = self.saliency_map_model.saliency_map(stimulus)
        saliency_map = self._prepare_saliency_map(saliency_map)
        log_density = self._f_log_density(saliency_map)
        return log_density

    def fit(self, stimuli, fixations, optimize=None):
        """
        Fit the parameters of the model
        """

        if optimize is None:
            optimize = ['blur_radius', 'nonlinearity', 'centerbias', 'alpha']

        x_inds = []
        y_inds = []
        for n in range(len(stimuli)):
            f = fixations[fixations.n == n]
            x_inds.append(f.x_int)
            y_inds.append(f.y_int)

        weights = np.array([len(inds) for inds in x_inds], dtype=float)
        weights /= weights.sum()

        smp = self.saliency_map_processing

        log_likelihood = smp.average_log_likelihood
        param_dict = {'blur_radius': smp.blur_radius,
                      'nonlinearity': smp.nonlinearity_ys,
                      'centerbias': smp.centerbias_ys,
                      'alpha': smp.alpha}
        params = [param_dict[name] for name in optimize]
        grads = T.grad(log_likelihood, params)

        print('Compiling theano function')
        sys.stdout.flush()
        f_ll_with_grad = theano.function([self.theano_input, self.saliency_map_processing.x_inds,
                                          self.saliency_map_processing.y_inds],
                                         [log_likelihood]+grads)

        print('Caching saliency maps')
        sys.stdout.flush()
        saliency_maps = []
        for s in stimuli:
            smap = self._prepare_saliency_map(self.saliency_map_model.saliency_map(s))
            saliency_maps.append(smap)

        full_params = self.saliency_map_processing.params

        def func(blur_radius, nonlinearity, centerbias, alpha, optimize=None):
            print('blur_radius: ', blur_radius)
            print('nonlinearity:', nonlinearity)
            print('centerbias:  ', centerbias)
            print('alpha:       ', alpha)
            for p, v in zip(full_params, [blur_radius, nonlinearity, centerbias, alpha]):
                p.set_value(v)

            values = []
            grads = [[] for p in params]
            for n in generics.progressinfo(range(len(stimuli))):
                rets = f_ll_with_grad(saliency_maps[n], x_inds[n], y_inds[n])
                values.append(rets[0])
                assert len(rets) == len(params)+1
                for l, g in zip(grads, rets[1:]):
                    l.append(g)

            value = np.average(values, axis=0, weights=weights)
            av_grads = []
            for grad in grads:
                av_grads.append(np.average(grad, axis=0, weights=weights))

            #print(value, av_grads)

            return value, tuple(av_grads)

        nonlinearity_value = self.saliency_map_processing.nonlinearity_ys.get_value()
        num_nonlinearity = len(nonlinearity_value)

        centerbias_value = self.saliency_map_processing.centerbias_ys.get_value()

        constraints = []
        constraints.append({'type': 'ineq',
                            'fun': lambda blur_radius, nonlinearity, centerbias, alpha: nonlinearity[0]})

        for i in range(1, num_nonlinearity):
            constraints.append({'type': 'ineq',
                                'fun': lambda blur_radius, nonlinearity, centerbias, alpha, i=i: nonlinearity[i] - nonlinearity[i-1]})

        constraints.append({'type': 'eq',
                            'fun': lambda blur_radius, nonlinearity, centerbias, alpha: nonlinearity.sum()-nonlinearity_value.sum()})
        constraints.append({'type': 'eq',
                            'fun': lambda blur_radius, nonlinearity, centerbias, alpha: centerbias.sum()-centerbias_value.sum()})

        options = {'disp': 2,
                   'iprint': 2,
                   'maxiter': 1000}

        x0 = {'blur_radius': full_params[0].get_value(),
              'nonlinearity': full_params[1].get_value(),
              'centerbias': full_params[2].get_value(),
              'alpha': full_params[3].get_value()}

        res = minimize(func, x0, jac=True, constraints=constraints, method='SLSQP', tol=1e-9, options=options, optimize=optimize)
        return res
