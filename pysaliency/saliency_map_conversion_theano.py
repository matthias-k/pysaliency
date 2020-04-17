# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function  # , unicode_literals

import sys

import numpy as np

from tqdm import tqdm


from .generics import progressinfo
from .models import Model, UniformModel
from .optpy import minimize
from .datasets import Fixations


def optimize_for_information_gain(
        model, fit_stimuli, fit_fixations,
        num_nonlinearity=20,
        num_centerbias=12,
        blur_radius=0,
        optimize=None,
        saliency_min=None,
        saliency_max=None,
        verbose=0,
        return_optimization_result=False,
        method='SLSQP',
        maxiter=1000):
    """ convert saliency map model into probabilistic model as described in Kümmerer et al, PNAS 2015.
    """

    if saliency_min is None or saliency_max is None:
        smax = -np.inf
        smin = np.inf
        for s in tqdm(fit_stimuli):
            smap = model.saliency_map(s)
            smax = np.max([smax, smap.max()])
            smin = np.min([smin, smap.min()])

        if saliency_min is None:
            saliency_min = smin
        if saliency_max is None:
            saliency_max = smax

    from .theano_utils import SaliencyMapProcessingLogNonlinearity
    processing_class = SaliencyMapProcessingLogNonlinearity
    nonlinearity_ys = np.linspace(-8, 0, num=num_nonlinearity)
    smc = SaliencyMapConvertor(model,
                               nonlinearity=nonlinearity_ys,
                               centerbias=np.linspace(1.1, 0.9, num=num_centerbias),
                               blur_radius=blur_radius,
                               processing_class=processing_class,
                               saliency_max=saliency_max,
                               saliency_min=saliency_min)
    res = smc.fit(fit_stimuli, fit_fixations, optimize=optimize, verbose=verbose, maxiter=maxiter, method=method)
    if return_optimization_result:
        return smc, res
    else:
        return smc


def optimize_saliency_map_conversion(saliency_map_processing, saliency_maps, x_inds, y_inds,
                                     baseline_model_loglikelihood, optimize=None, verbose=0, method='SLSQP',
                                     nonlinearity_min=1e-8,
                                     view=None,
                                     tol=None,
                                     maxiter=1000):
    """
    Fit the parameters of the model

    Parameters
    ----------

    @type  verbose: int
    @param verbose: controls the verbosity of the output. Possible values:
                    0: No output at all
                    1: Output optimization progress (given in bits/fix relative to baseline)
                    2: Additionally output current parameters
                    3: Additionally give feedback on evaluation

    @type  baseline_model: GenericModel
    @param baseline_model: Output optimization progress relative to this model.
                            The default is a uniform model.

    @param view: a ipython cluster view to parallelize the gradients over
                 multiple stimuli
    """

    import theano
    import theano.tensor as T
    from .theano_utils import SaliencyMapProcessing, SaliencyMapProcessingLogNonlinearity, SaliencyMapProcessingLogarithmic

    if optimize is None:
        optimize = ['blur_radius', 'nonlinearity', 'centerbias', 'alpha']

    weights = np.array([len(inds) for inds in x_inds], dtype=theano.config.floatX)
    weights /= weights.sum()
    x_inds = [x.copy() for x in x_inds]
    y_inds = [y.copy() for y in y_inds]

    if True:
        smp = saliency_map_processing

        negative_log_likelihood = -smp.average_log_likelihood / np.log(2)
        param_dict = {'blur_radius': smp.blur_radius,
                      'nonlinearity': smp.nonlinearity_ys,
                      'centerbias': smp.centerbias_ys,
                      'alpha': smp.alpha}
        params = [param_dict[name] for name in optimize]
        grads = T.grad(negative_log_likelihood, params)

        print('Compiling theano function')
        sys.stdout.flush()

        full_params = smp.params

        baseline = baseline_model_loglikelihood / np.log(2)
    if view is not None:
        dv = view.client[:]
        dv.execute('import numpy as np; import theano; import theano.tensor as T', block=True)

        def build_function(processing_class, blur_radius, nonlinearity, centerbias, alpha, nonlinearity_xs, optimize):
            global cost_with_grads
            global full_params
            theano_input = T.matrix('saliency_map', dtype=theano.config.floatX)
            smp = processing_class(theano_input,
                                   sigma=blur_radius,
                                   nonlinearity_ys=nonlinearity,
                                   centerbias=centerbias,
                                   alpha=alpha
                                   )
            smp.theano_objects.nonlinearity.nonlinearity_xs.set_value(np.array(nonlinearity_xs, dtype=theano.config.floatX))
            negative_log_likelihood = -smp.average_log_likelihood / np.log(2)
            param_dict = {'blur_radius': smp.blur_radius,
                          'nonlinearity': smp.nonlinearity_ys,
                          'centerbias': smp.centerbias_ys,
                          'alpha': smp.alpha}
            params = [param_dict[name] for name in optimize]
            grads = T.grad(negative_log_likelihood, params)

            print('Compiling theano function')
            sys.stdout.flush()
            f_ll_with_grad = theano.function([smp.saliency_map, smp.x_inds,
                                              smp.y_inds],
                                             [negative_log_likelihood] + grads)
            cost_with_grads = f_ll_with_grad
            full_params = smp.params

        def set_param_clients(blur_radius, nonlinearity, centerbias, alpha):
            global full_params
            for p, v in zip(full_params, [blur_radius, nonlinearity, centerbias, alpha]):
                p.set_value(v.astype(p.dtype))

        print("Building theano functions in clients")
        dv.apply(build_function, type(saliency_map_processing),
                 saliency_map_processing.blur_radius.get_value(),
                 saliency_map_processing.nonlinearity_ys.get_value(),
                 saliency_map_processing.centerbias_ys.get_value(),
                 saliency_map_processing.alpha.get_value(),
                 saliency_map_processing.theano_objects.nonlinearity.nonlinearity_xs.get_value(),
                 optimize).wait_interactive()
    else:
        f_ll_with_grad = theano.function([smp.saliency_map, smp.x_inds,
                                          smp.y_inds],
                                         [negative_log_likelihood] + grads)

    def func(blur_radius, nonlinearity, centerbias, alpha, optimize=None):
        if verbose > 1:
            print('blur_radius: ', blur_radius)
            print('nonlinearity:', nonlinearity)
            print('centerbias:  ', centerbias)
            print('alpha:       ', alpha)

        for p, v in zip(full_params, [blur_radius, nonlinearity, centerbias, alpha]):
            p.set_value(v.astype(p.dtype))

        values = []
        grads = [[] for p in params]
        all_rets = []
        if view is None:
            for n in progressinfo(range(len(saliency_maps)), verbose=verbose > 2):
                rets = f_ll_with_grad(saliency_maps[n], x_inds[n], y_inds[n])
                all_rets.append(rets)
        else:
            def f(smap, x, y):
                global cost_with_grads
                return cost_with_grads(smap, x.copy(), y.copy())
            dv.apply(set_param_clients, blur_radius, nonlinearity, centerbias, alpha).wait()
            async_rets = view.map_async(f, saliency_maps, x_inds, y_inds)
            async_rets.wait_interactive()
            all_rets = list(async_rets)

        for n in progressinfo(range(len(saliency_maps)), verbose=verbose > 10):
            if len(x_inds[n]):
                rets = all_rets[n]
                values.append(rets[0])
                assert len(rets) == len(params) + 1
                for l, g in zip(grads, rets[1:]):
                    l.append(np.array(g))
            else:
                # No fixations for this image. The theano functions will return
                # NaN which would screw up the weighted average
                values.append(0.0)
                for l, p in zip(grads, params):
                    l.append(np.zeros_like(p.get_value()))

        value = np.average(values, axis=0, weights=weights)
        value += baseline
        av_grads = []
        for grad in grads:
            av_grads.append(np.average(grad, axis=0, weights=weights))

        return value, tuple(av_grads)

    nonlinearity_value = smp.nonlinearity_ys.get_value()
    num_nonlinearity = len(nonlinearity_value)

    centerbias_value = smp.centerbias_ys.get_value()

    if isinstance(saliency_map_processing, SaliencyMapProcessing):
        print("Using density constraints")
        bounds = {'nonlinearity': [(nonlinearity_min, 1e7) for i in range(num_nonlinearity)],
                  'centerbias': [(1e-7, 1000) for i in range(len(centerbias_value))],
                  'alpha': [(1e-4, 1e4)],
                  'blur_radius': [(0.0, 1e3)]}

        constraints = []

        if 'nonlineariy' in optimize:
            for i in range(1, num_nonlinearity):
                constraints.append({'type': 'ineq',
                                   'fun': lambda blur_radius, nonlinearity, centerbias, alpha, i=i: nonlinearity[i] - nonlinearity[i - 1]})

            constraints.append({'type': 'eq',
                                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: nonlinearity.sum() - nonlinearity_value.sum()})
        if 'centerbias' in optimize:
            constraints.append({'type': 'eq',
                                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: centerbias.sum() - centerbias_value.sum()})
    if isinstance(saliency_map_processing, SaliencyMapProcessingLogNonlinearity):
        print("Using density constraints on log scale")
        bounds = {'nonlinearity': [(-100, 100) for i in range(num_nonlinearity)],
                  'centerbias': [(1e-7, 1000) for i in range(len(centerbias_value))],
                  'alpha': [(1e-4, 1e4)],
                  'blur_radius': [(0.0, 1e3)]}

        constraints = []

        if 'nonlinearity' in optimize:
            for i in range(1, num_nonlinearity):
                constraints.append({'type': 'ineq',
                                    'fun': lambda blur_radius, nonlinearity, centerbias, alpha, i=i: nonlinearity[i] - nonlinearity[i - 1]})

            constraints.append({'type': 'eq',
                                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: np.exp(nonlinearity).sum() - np.exp(nonlinearity_value).sum()})
        if 'centerbias' in optimize:
            constraints.append({'type': 'eq',
                                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: centerbias.sum() - centerbias_value.sum()})

    elif isinstance(saliency_map_processing, SaliencyMapProcessingLogarithmic):
        print("Using log density constraints")
        bounds = {'nonlinearity': [(None, None) for i in range(num_nonlinearity)],
                  'centerbias': [(None, None) for i in range(len(centerbias_value))],
                  'alpha': [(1e-4, 1e4)],
                  'blur_radius': [(0.0, 1e3)]}

        constraints = []
        if 'nonlinearity' in optimize:
            constraints.append({'type': 'eq',
                                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: nonlinearity[0]})
            for i in range(1, num_nonlinearity):
                constraints.append({'type': 'ineq',
                                    'fun': lambda blur_radius, nonlinearity, centerbias, alpha, i=i: nonlinearity[i] - nonlinearity[i - 1]})

        if 'centerbias' in optimize:
            constraints.append({'type': 'eq',
                                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: centerbias[0]})
    else:
        raise TypeError('Unknown processing class', saliency_map_processing)

    if method == 'SLSQP':
        options = {'iprint': 2, 'disp': 2, 'maxiter': maxiter,
                   'eps': 1e-9
                   }
        tol = tol or 1e-9
    elif method == 'IPOPT':
        tol = tol or 1e-7
        options = {'disp': 5, 'maxiter': maxiter,
                   'tol': tol
                   }
    else:
        options = {'maxiter': maxiter}

    x0 = {'blur_radius': full_params[0].get_value(),
          'nonlinearity': full_params[1].get_value(),
          'centerbias': full_params[2].get_value(),
          'alpha': full_params[3].get_value()}

    res = minimize(func, x0, jac=True, constraints=constraints, bounds=bounds, method=method, tol=tol, options=options, optimize=optimize)
    return res


class SaliencyMapConvertor(Model):
    """
    Convert saliency map models to probabilistic models.
    """
    def __init__(self, saliency_map_model, nonlinearity=None, centerbias=None, alpha=1.0, blur_radius=0,
                 saliency_min=None, saliency_max=None, cache_location=None, processing_class=None):
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
        from .theano_utils import SaliencyMapProcessing, SaliencyMapProcessingLogNonlinearity

        if processing_class is None:
            processing_class = SaliencyMapProcessing

        self.saliency_map_model = saliency_map_model
        if nonlinearity is None:
            nonlinearity = np.linspace(0, 1.0, num=20)
        if centerbias is None:
            if processing_class == SaliencyMapProcessing or processing_class == SaliencyMapProcessingLogNonlinearity:
                centerbias = np.ones(12)
            else:
                centerbias = np.zeros(12)

        nonlinearity = np.asarray(nonlinearity)
        centerbias = np.asarray(centerbias)

        self._blur_radius = blur_radius
        self._nonlinearity = nonlinearity
        self._centerbias = centerbias
        self._alpha = alpha

        self.saliency_min = saliency_min
        self.saliency_max = saliency_max

        if processing_class is None:
            processing_class = SaliencyMapProcessingLogNonlinearity
        self.processing_class = processing_class

        self._build()

    def set_params(self, **kwargs):
        import theano
        no_of_kwargs = len(kwargs)
        if 'nonlinearity' in kwargs:
            self.saliency_map_processing.nonlinearity_ys.set_value(kwargs.pop('nonlinearity').astype(theano.config.floatX))
        if 'centerbias' in kwargs:
            self.saliency_map_processing.centerbias_ys.set_value(kwargs.pop('centerbias').astype(theano.config.floatX))
        if 'alpha' in kwargs:
            self.saliency_map_processing.alpha.set_value(np.array(kwargs.pop('alpha'), dtype=theano.config.floatX))
        if 'blur_radius' in kwargs:
            self.saliency_map_processing.blur_radius.set_value(np.array(kwargs.pop('blur_radius'), theano.config.floatX))
        if 'saliency_min' in kwargs:
            self.saliency_min = kwargs.pop('saliency_min')
        if 'saliency_max' in kwargs:
            self.saliency_max = kwargs.pop('saliency_max')

        if no_of_kwargs != len(kwargs):
            # We used some keywords, thus we have to clear the cache
            self._cache.clear()

        super(SaliencyMapConvertor, self).set_params(**kwargs)

    def _build(self):
        import theano
        import theano.tensor as T
        self.theano_input = T.matrix('saliency_map', dtype=theano.config.floatX)
        self.saliency_map_processing = self.processing_class(self.theano_input,
                                                             sigma=self._blur_radius,
                                                             nonlinearity_ys=self._nonlinearity,
                                                             centerbias=self._centerbias,
                                                             alpha=self._alpha
                                                             )
        self._f_log_density = theano.function([self.theano_input], self.saliency_map_processing.log_density,
                                              allow_input_downcast=True)

    def _prepare_saliency_map(self, saliency_map):
        import theano
        smin, smax = self.saliency_min, self.saliency_max
        if smin is None:
            smin = saliency_map.min()
        if smax is None:
            smax = saliency_map.max()

        saliency_map = (saliency_map - smin) / (smax - smin)
        saliency_map = saliency_map.astype(theano.config.floatX)
        return saliency_map

    def _log_density(self, stimulus):
        saliency_map = self.saliency_map_model.saliency_map(stimulus)
        saliency_map = self._prepare_saliency_map(saliency_map)
        log_density = self._f_log_density(saliency_map)
        return log_density

    def fit(self, stimuli, fixations, optimize=None, verbose=0, baseline_model=None, method='SLSQP',
            nonlinearity_min=1e-8,
            view=None, tol=None,
            maxiter=1000):
        """
        Fit the parameters of the model

        Parameters
        ----------

        @type  verbose: int
        @param verbose: controls the verbosity of the output. Possible values:
                        0: No output at all
                        1: Output optimization progress (given in bits/fix relative to baseline)
                        2: Additionally output current parameters
                        3: Additionally give feedback on evaluation

        @type  baseline_model: GenericModel
        @param baseline_model: Output optimization progress relative to this model.
                               The default is a uniform model.
        """
        print('Caching saliency maps')
        sys.stdout.flush()
        saliency_maps = []
        for n, s in enumerate(progressinfo(stimuli, verbose=verbose > 1)):
            smap = self._prepare_saliency_map(self.saliency_map_model.saliency_map(s))
            saliency_maps.append(smap)
        self.saliency_map_model._cache.clear()

        x_inds = []
        y_inds = []
        for n in tqdm(list(range(len(stimuli))), desc='Preparing fixations', disable=verbose < 2):
            inds = fixations.n == n
            x_inds.append(fixations.x_int[inds].copy())
            y_inds.append(fixations.y_int[inds].copy())

        if baseline_model is None:
            baseline_model = UniformModel()
        baseline = baseline_model.log_likelihood(stimuli, fixations)

        res = optimize_saliency_map_conversion(self.saliency_map_processing,
                                               saliency_maps, x_inds, y_inds, baseline,
                                               optimize=optimize, verbose=verbose, method=method,
                                               nonlinearity_min=nonlinearity_min,
                                               view=view,
                                               tol=tol,
                                               maxiter=maxiter)

        self.set_params(nonlinearity=res.nonlinearity, centerbias=res.centerbias, alpha=res.alpha, blur_radius=res.blur_radius)
        return res

    def __getstate__(self):
        self._nonlinearity = self.saliency_map_processing.nonlinearity_ys.get_value()
        self._centerbias = self.saliency_map_processing.centerbias_ys.get_value()
        self._alpha = self.saliency_map_processing.alpha.get_value()
        self._blur_radius = self.saliency_map_processing.blur_radius.get_value()
        state = dict(self.__dict__)
        del state['saliency_map_processing']
        del state['theano_input']
        del state['_f_log_density']
        return state

    def __setstate__(self, state):
        self.__dict__ = dict(state)
        self._build()


class JointSaliencyMapConvertor(object):
    def __init__(self, saliency_map_models, nonlinearity=None, centerbias=None, alpha=1.0, blur_radius=0,
                 saliency_min=None, saliency_max=None, cache_location=None, processing_class=None):
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
        self.saliency_map_models = saliency_map_models
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

        from .theano_utils import SaliencyMapProcessingLogNonlinearity
        if processing_class is None:
            processing_class = SaliencyMapProcessingLogNonlinearity
        self.processing_class = processing_class

        self._build()

    def set_params(self, **kwargs):
        if 'nonlinearity' in kwargs:
            self.saliency_map_processing.nonlinearity_ys.set_value(kwargs.pop('nonlinearity'))
        if 'centerbias' in kwargs:
            self.saliency_map_processing.centerbias_ys.set_value(kwargs.pop('centerbias'))
        if 'alpha' in kwargs:
            self.saliency_map_processing.alpha.set_value(kwargs.pop('alpha'))
        if 'blur_radius' in kwargs:
            self.saliency_map_processing.blur_radius.set_value(kwargs.pop('blur_radius'))
        if 'saliency_min' in kwargs:
            self.saliency_min = kwargs.pop('saliency_min')
        if 'saliency_max' in kwargs:
            self.saliency_max = kwargs.pop('saliency_max')

    def _build(self):
        import theano
        import theano.tensor as T
        self.theano_input = T.matrix('saliency_map', dtype=theano.config.floatX)
        self.saliency_map_processing = self.processing_class(self.theano_input,
                                                             sigma=self._blur_radius,
                                                             nonlinearity_ys=self._nonlinearity,
                                                             centerbias=self._centerbias,
                                                             alpha=self._alpha
                                                             )
        self._f_log_density = theano.function([self.theano_input], self.saliency_map_processing.log_density)

    def _prepare_saliency_map(self, saliency_map):
        import theano
        smin, smax = self.saliency_min, self.saliency_max
        if smin is None:
            smin = saliency_map.min()
        if smax is None:
            smax = saliency_map.max()

        saliency_map = (saliency_map - smin) / (smax - smin)
        saliency_map = saliency_map.astype(theano.config.floatX)
        return saliency_map

    def fit(self, stimuli, fixations, optimize=None, verbose=0, baseline_model=None, method='SLSQP',
            nonlinearity_min=1e-8, view=None, tol=None, maxiter=1000):
        """
        Fit the parameters of the model

        Parameters
        ----------

        @type  verbose: int
        @param verbose: controls the verbosity of the output. Possible values:
                        0: No output at all
                        1: Output optimization progress (given in bits/fix relative to baseline)
                        2: Additionally output current parameters
                        3: Additionally give feedback on evaluation

        @type  baseline_model: GenericModel
        @param baseline_model: Output optimization progress relative to this model.
                               The default is a uniform model.
        """
        if isinstance(fixations, Fixations):
            fixations = [fixations for m in self.saliency_map_models]

        assert len(fixations) == len(self.saliency_map_models)

        print('Caching saliency maps')
        sys.stdout.flush()
        saliency_maps = []
        x_inds = []
        y_inds = []
        for n, s in enumerate(progressinfo(stimuli, verbose=verbose > 1)):
            for saliency_map_model, f in zip(self.saliency_map_models, fixations):
                smap = self._prepare_saliency_map(saliency_map_model.saliency_map(s))
                saliency_maps.append(smap)
                ff = f[f.n == n]
                x_inds.append(ff.x_int)
                y_inds.append(ff.y_int)
                saliency_map_model._cache.clear()

        if baseline_model is None:
            baseline_model = UniformModel()
        lls = []
        for f in fixations:
            lls.append(baseline_model.log_likelihood(stimuli, f))
        lls = np.hstack(lls)
        baseline = np.mean(lls)

        res = optimize_saliency_map_conversion(self.saliency_map_processing,
                                               saliency_maps, x_inds, y_inds, baseline,
                                               optimize=optimize, verbose=verbose, method=method,
                                               nonlinearity_min=nonlinearity_min,
                                               view=view,
                                               tol=tol,
                                               maxiter=maxiter)

        self.set_params(nonlinearity=res.nonlinearity, centerbias=res.centerbias, alpha=res.alpha, blur_radius=res.blur_radius)
        return res

    def __getstate__(self):
        self._nonlinearity = self.saliency_map_processing.nonlinearity_ys.get_value()
        self._centerbias = self.saliency_map_processing.centerbias_ys.get_value()
        self._alpha = self.saliency_map_processing.alpha.get_value()
        self._blur_radius = self.saliency_map_processing.blur_radius.get_value()
        state = dict(self.__dict__)
        del state['saliency_map_processing']
        del state['theano_input']
        del state['_f_log_density']
        return state

    def __setstate__(self, state):
        self.__dict__ = dict(state)
        self._build()
