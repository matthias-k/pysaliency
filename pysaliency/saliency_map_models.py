from __future__ import absolute_import, print_function, division, unicode_literals

import os
from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave
from scipy.ndimage import gaussian_filter, zoom

from tqdm import tqdm
from boltons.cacheutils import cached, LRU

from .generics import progressinfo
from .roc import general_roc, general_rocs_per_positive

from .utils import TemporaryDirectory, run_matlab_cmd, Cache
from .datasets import Stimulus, Fixations, FileStimuli


def handle_stimulus(stimulus):
    """
    Make sure that a stimulus is a `Stimulus`-object
    """
    if not isinstance(stimulus, Stimulus):
        stimulus = Stimulus(stimulus)
    return stimulus


def normalize_saliency_map(saliency_map, cdf, cdf_bins):
    """ Normalize saliency to make saliency values distributed according to a given CDF
    """

    smap = saliency_map.copy()
    shape = smap.shape
    smap = smap.flatten()
    smap = np.argsort(np.argsort(smap)).astype(float)
    smap /= 1.0*len(smap)

    inds = np.searchsorted(cdf, smap, side='right')
    smap = cdf_bins[inds]
    smap = smap.reshape(shape)
    smap = smap.reshape(shape)
    return smap


class FullShuffledNonfixationProvider(object):
    def __init__(self, stimuli, fixations, max_fixations_in_cache=500*1000*1000):
        self.stimuli = stimuli
        self.fixations = fixations
        cache_size = int(max_fixations_in_cache / len(self.fixations.x))
        self.cache = LRU(cache_size)
        self.nonfixations_for_image = cached(self.cache)(self._nonfixations_for_image)
        self.widths = np.asarray([s[1] for s in stimuli.sizes]).astype(float)
        self.heights = np.asarray([s[0] for s in stimuli.sizes]).astype(float)

    def _nonfixations_for_image(self, n):
        inds = ~(self.fixations.n == n)
        xs = (self.fixations.x[inds].copy()).astype(float)
        ys = (self.fixations.y[inds].copy()).astype(float)

        other_ns = self.fixations.n[inds]
        xs *= self.stimuli.sizes[n][1]/self.widths[other_ns]
        ys *= self.stimuli.sizes[n][0]/self.heights[other_ns]

        return xs.astype(int), ys.astype(int)

    def __call__(self, stimuli, fixations, i):
        assert stimuli is self.stimuli
        n = fixations.n[i]
        return self.nonfixations_for_image(n)


@add_metaclass(ABCMeta)
class GeneralSaliencyMapModel(object):
    """
    Most general saliency model class. The model is neither
    assumed to be time-independet nor to be a probabilistic
    model.
    """

    @abstractmethod
    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist, out=None):
        """
        Return the models saliency map prediction depending on a fixation history
        for the n-th image.
        """
        raise NotImplementedError()

    def AUCs(self, stimuli, fixations, nonfixations='uniform', verbose=False):
        """
        Calulate AUC scores for fixations

        :type fixations : Fixations
        :param fixations : Fixation object to calculate the AUC scores for.

        :type nonfixations : string or Fixations
        :param nonfixations : Nonfixations to use for calculating AUC scores.
                              Possible values are:
                                  'uniform':  Use uniform nonfixation distribution (Judd-AUC), i.e.
                                              all pixels from the saliency map.
                                  'shuffled': Use all fixations from other images as nonfixations.
                                  fixations-object: For each image, use the fixations in this fixation
                                                    object as nonfixations

        :rtype : ndarray
        :return : list of AUC scores for each fixation,
                  ordered as in `fixations.x` (average=='fixation' or None)
                  or by image numbers (average=='image')
        """
        rocs_per_fixation = []
        rocs = {}
        out = None

        nonfix_ys = None
        nonfix_xs = None

        if isinstance(nonfixations, Fixations):
            nonfix_xs = []
            nonfix_ys = []
            for n in range(fixations.n.max()+1):
                inds = nonfixations.n == n
                nonfix_xs.append(nonfixations.x_int[inds].copy())
                nonfix_ys.append(nonfixations.y_int[inds].copy())

        if nonfixations == 'shuffled':
            nonfixations = FullShuffledNonfixationProvider(stimuli, fixations)

        for i in tqdm(range(len(fixations.x)), total=len(fixations.x), disable=not verbose):
            out = self.conditional_saliency_map(stimuli.stimulus_objects[fixations.n[i]], fixations.x_hist[i], fixations.y_hist[i],
                                                fixations.t_hist[i], out=out)
            positives = np.asarray([out[fixations.y_int[i], fixations.x_int[i]]])
            if nonfixations == 'uniform':
                negatives = out.flatten()
            elif nonfix_xs is not None:
                n = fixations.n[i]
                negatives = out[nonfix_ys[n], nonfix_xs[n]]
            else:
                _nonfix_xs, _nonfix_ys = nonfixations(stimuli, fixations, i)
                negatives = out[_nonfix_ys.astype(int), _nonfix_xs.astype(int)]
            this_roc, _, _ = general_roc(positives, negatives)
            rocs.setdefault(fixations.n[i], []).append(this_roc)
            rocs_per_fixation.append(this_roc)
        return np.asarray(rocs_per_fixation)

    def AUC(self, stimuli, fixations, nonfixations='uniform', average='fixation', verbose=False):
        """
        Calulate AUC scores for fixations

        :type fixations : Fixations
        :param fixations : Fixation object to calculate the AUC scores for.

        :type nonfixations : string or Fixations
        :param nonfixations : Nonfixations to use for calculating AUC scores.
                              Possible values are:
                                  'uniform':  Use uniform nonfixation distribution (Judd-AUC), i.e.
                                              all pixels from the saliency map.
                                  'shuffled': Use all fixations from other images as nonfixations.
                                  fixations-object: For each image, use the fixations in this fixation
                                                    object as nonfixations

        :type average : string
        :param average : How to average the AUC scores for each fixation.
                         Possible values are:
                             'image': average over images
                             'fixation' or None: Return AUC score for each fixation separately

        :rtype : ndarray
        :return : list of AUC scores for each fixation,
                  ordered as in `fixations.x` (average=='fixation' or None)
                  or by image numbers (average=='image')
        """
        if average != 'fixation':
            raise NotImplementedError()
        aucs = self.AUCs(stimuli, fixations, nonfixations=nonfixations, verbose=verbose)
        return np.mean(aucs)

    def NSSs(self, stimuli, fixations, verbose=False):
        values = np.empty(len(fixations.x))
        out = None

        for i in tqdm(range(len(fixations.x)), disable=not verbose, total=len(fixations.x)):
            out = self.conditional_saliency_map(stimuli.stimulus_objects[fixations.n[i]], fixations.x_hist[i], fixations.y_hist[i],
                                                fixations.t_hist[i], out=out)

            smap = out
            mean = smap.mean()
            std = smap.std()

            value = smap[fixations.y_int[i], fixations.x_int[i]].copy()
            value -= mean
            value /= std

            values[i] = value
        return values

    def NSS(self, stimuli, fixations, verbose=False):
        return self.NSSs(stimuli, fixations, verbose=verbose).mean()

    def set_params(self, **kwargs):
        """
        Set model parameters, if the model has parameters

        This method has to reset caches etc., if the depend on the parameters
        """
        if kwargs:
            raise ValueError('Unkown parameters!', kwargs)


class SaliencyMapModel(GeneralSaliencyMapModel):
    """
    Most model class for saliency maps. The model is assumed
    to be stationary in time (i.e. all fixations are independent)
    but the model is not explicitly a probabilistic model.
    """

    def __init__(self, cache_location = None, caching=True,
                 memory_cache_size=None):
        self._cache = Cache(cache_location, memory_cache_size=memory_cache_size)
        self.caching = caching

    @property
    def cache_location(self):
        return self._cache.cache_location

    @cache_location.setter
    def cache_location(self, value):
        self._cache.cache_location = value

    def saliency_map(self, stimulus):
        """
        Get saliency map for given stimulus.

        To overwrite this function, overwrite `_saliency_map` as otherwise
        the caching mechanism is disabled.
        """
        stimulus = handle_stimulus(stimulus)
        if not self.caching:
            return self._saliency_map(stimulus.stimulus_data)
        stimulus_id = stimulus.stimulus_id
        if not stimulus_id in self._cache:
            self._cache[stimulus_id] = self._saliency_map(stimulus.stimulus_data)
        return self._cache[stimulus_id]

    @abstractmethod
    def _saliency_map(self, stimulus):
        """
        Overwrite this to implement you own SaliencyMapModel.

        Parameters
        ----------

        @type  stimulus: ndarray
        @param stimulus: stimulus for which the saliency map should be computed.
        """
        raise NotImplementedError()

    def conditional_saliency_map(self, stimulus, *args, **kwargs):
        return self.saliency_map(stimulus)

    def AUCs(self, stimuli, fixations, nonfixations='uniform', verbose=False):
        """
        Calulate AUC scores for fixations

        :type fixations : Fixations
        :param fixations : Fixation object to calculate the AUC scores for.

        :type nonfixations : string or Fixations
        :param nonfixations : Nonfixations to use for calculating AUC scores.
                              Possible values are:
                                  'uniform':  Use uniform nonfixation distribution (Judd-AUC), i.e.
                                              all pixels from the saliency map.
                                  'shuffled': Use all fixations from other images as nonfixations.
                                  fixations-object: For each image, use the fixations in this fixation
                                                    object as nonfixations

        :rtype : ndarray
        :return : list of AUC scores for each fixation,
                  ordered as in `fixations.x` (average=='fixation' or None)
                  or by image numbers (average=='image')
        """
        rocs_per_fixation = np.empty(len(fixations.x))

        nonfix_ys = None
        nonfix_xs = None

        if isinstance(nonfixations, Fixations):
            nonfix_xs = []
            nonfix_ys = []
            for n in range(fixations.n.max()+1):
                inds = nonfixations.n == n
                nonfix_xs.append(nonfixations.x_int[inds].copy())
                nonfix_ys.append(nonfixations.y_int[inds].copy())

        if nonfixations == 'shuffled':
            nonfixations = FullShuffledNonfixationProvider(stimuli, fixations)

        for n in tqdm(range(len(stimuli)), total=len(stimuli), disable = not verbose):
            inds = fixations.n == n
            if not inds.sum():
                continue
            out = self.saliency_map(stimuli.stimulus_objects[n])
            positives = np.asarray(out[fixations.y_int[inds], fixations.x_int[inds]])
            if nonfixations == 'uniform':
                negatives = out.flatten()
            elif nonfix_xs is not None:
                negatives = out[nonfix_ys[n], nonfix_xs[n]]
            else:
                _nonfix_xs, _nonfix_ys = nonfixations(stimuli, fixations, np.nonzero(inds)[0][0])
                negatives = out[_nonfix_ys.astype(int), _nonfix_xs.astype(int)]

            positives = positives.astype(float)
            negatives = negatives.astype(float)

            rocs = general_rocs_per_positive(positives, negatives)
            rocs_per_fixation[inds] = rocs

        return rocs_per_fixation

    def AUC_per_image(self, stimuli, fixations, nonfixations='uniform', verbose=False):
        """
        Calulate AUC scores per image for fixations

        :type fixations : Fixations
        :param fixations : Fixation object to calculate the AUC scores for.

        :type nonfixations : string or Fixations
        :param nonfixations : Nonfixations to use for calculating AUC scores.
                              Possible values are:
                                  'uniform':  Use uniform nonfixation distribution (Judd-AUC), i.e.
                                              all pixels from the saliency map.
                                  'shuffled': Use all fixations from other images as nonfixations.
                                  fixations-object: For each image, use the fixations in this fixation
                                                    object as nonfixations

        :rtype : ndarray
        :return : list of AUC scores for each image,
                  or by image numbers (average=='image')
        """
        rocs_per_image = []
        out = None

        nonfix_xs = None
        nonfix_ys = None

        if isinstance(nonfixations, Fixations):
            nonfix_xs = []
            nonfix_ys = []
            for n in range(fixations.n.max()+1):
                inds = nonfixations.n == n
                nonfix_xs.append(nonfixations.x_int[inds].copy())
                nonfix_ys.append(nonfixations.y_int[inds].copy())

        if nonfixations == 'shuffled':
            nonfixations = FullShuffledNonfixationProvider(stimuli, fixations)

        for n in progressinfo(range(len(stimuli)), verbose=verbose):
            out = self.saliency_map(stimuli.stimulus_objects[n])
            inds = fixations.n == n
            positives = np.asarray(out[fixations.y_int[inds], fixations.x_int[inds]])
            if nonfixations == 'uniform':
                negatives = out.flatten()
            elif nonfix_xs is not None:
                negatives = out[nonfix_ys[n], nonfix_xs[n]]
            else:
                _nonfix_xs, _nonfix_ys = nonfixations(stimuli, fixations, np.nonzero(inds)[0][0])
                negatives = out[_nonfix_ys.astype(int), _nonfix_xs.astype(int)]

            positives = positives.astype(float)
            negatives = negatives.astype(float)
            this_roc, _, _ = general_roc(positives, negatives)
            rocs_per_image.append(this_roc)
        return rocs_per_image

    def AUC(self, stimuli, fixations, nonfixations='uniform', average='fixation', verbose=False):
        """
        Calulate AUC scores for fixations

        :type fixations : Fixations
        :param fixations : Fixation object to calculate the AUC scores for.

        :type nonfixations : string or Fixations
        :param nonfixations : Nonfixations to use for calculating AUC scores.
                              Possible values are:
                                  'uniform':  Use uniform nonfixation distribution (Judd-AUC), i.e.
                                              all pixels from the saliency map.
                                  'shuffled': Use all fixations from other images as nonfixations.
                                  fixations-object: For each image, use the fixations in this fixation
                                                    object as nonfixations

        :type average : string
        :param average : How to average the AUC scores for each fixation.
                         Possible values are:
                             'image': average over images
                             'fixation' or None: Return AUC score for each fixation separately

        :rtype : ndarray
        :return : list of AUC scores for each fixation,
                  ordered as in `fixations.x` (average=='fixation' or None)
                  or by image numbers (average=='image')
        """
        if average not in ['fixation', 'image']:
            raise NotImplementedError()
        aucs = np.asarray(self.AUC_per_image(stimuli, fixations, nonfixations=nonfixations, verbose=verbose))
        if average == 'fixation':
            weights = np.zeros_like(aucs)
            for n in set(fixations.n):
                weights[n] = (fixations.n == n).mean()
            weights /= weights.sum()

            # take care of nans due to no fixations
            aucs[weights == 0] = 0

            return np.average(aucs, weights=weights)
        elif average == 'image':
            return np.mean(aucs)
        else:
            raise ValueError(average)

    def fixation_based_KL_divergence(self, stimuli, fixations, nonfixations='shuffled', bins=10, eps=1e-20):
        """
        Calulate fixation-based KL-divergences for fixations

        :type fixations : Fixations
        :param fixations : Fixation object to calculate the AUC scores for.

        :type nonfixations : string or Fixations
        :param nonfixations : Nonfixations to use for calculating AUC scores.
                              Possible values are:
                                  'uniform':  Use uniform nonfixation distribution (Judd-AUC), i.e.
                                              all pixels from the saliency map.
                                  'shuffled': Use all fixations from other images as nonfixations.
                                  fixations-object: For each image, use the fixations in this fixation
                                                    object as nonfixations

        :type  bins : int
        :param bins : Number of bins to use in estimating the fixation based KL divergence

        :type  eps : float
        :param eps : regularization constant for the KL divergence to avoid logarithms of zero.


        :rtype : float
        :return : fixation based KL divergence
        """

        fixation_values = []
        nonfixation_values = []

        saliency_min = np.inf
        saliency_max = -np.inf

        for n in range(len(stimuli.stimuli)):
            saliency_map = self.saliency_map(stimuli.stimulus_objects[n])
            saliency_min = min(saliency_min, saliency_map.min())
            saliency_max = max(saliency_max, saliency_map.max())

            f = fixations[fixations.n == n]
            fixation_values.append(saliency_map[f.y_int, f.x_int])
            if nonfixations == 'uniform':
                nonfixation_values.append(saliency_map.flatten())
            elif nonfixations == 'shuffled':
                f = fixations[fixations.n != n]
                widths = np.asarray([s[1] for s in stimuli.sizes]).astype(float)
                heights = np.asarray([s[0] for s in stimuli.sizes]).astype(float)
                xs = (f.x.copy())
                ys = (f.y.copy())
                other_ns = f.n

                xs *= stimuli.sizes[n][1]/widths[other_ns]
                ys *= stimuli.sizes[n][0]/heights[other_ns]

                nonfixation_values.append(saliency_map[ys.astype(int), xs.astype(int)])
            else:
                nonfix = nonfixations[nonfixations.n == n]
                nonfixation_values.append(saliency_map[nonfix.y_int, nonfix.x_int])

        fixation_values = np.hstack(fixation_values)
        nonfixation_values = np.hstack(nonfixation_values)

        hist_range = saliency_min, saliency_max

        p_fix, _ = np.histogram(fixation_values, bins=bins, range=hist_range, density=True)
        p_fix += eps
        p_fix /= p_fix.sum()
        p_nonfix, _ = np.histogram(nonfixation_values, bins=bins, range=hist_range, density=True)
        p_nonfix += eps
        p_nonfix /= p_nonfix.sum()

        return (p_fix * (np.log(p_fix) - np.log(p_nonfix))).sum()

    def image_based_kl_divergences(self, stimuli, gold_standard, minimum_value=1e-20, convert_gold_standard=True):
        """Calculate image-based KL-Divergences between model and gold standard for each stimulus

        This metric converts the model to a probabilistic saliency model by treating the
        saliency maps as probability densities (compare Wilming et al.) and calculates
        the KL Divergences between model and gold standard per stimulus.

        If the gold standard is already a probabilistic model that should not be converted in a
        new (different!) probabilistic model, set `convert_gold_standard` to False.
        """
        def convert_model(model, minimum_value):
            model_min = np.inf
            model_max = -np.inf
            for s in stimuli:
                smap = model.saliency_map(s)
                model_min = min(model_min, smap.min())
                model_max = max(model_max, smap.max())
            new_min = model_min / model_max
            if new_min == 1.0:
                # constant saliency map model
                new_min = 0.0
            new_min = max(new_min, minimum_value)
            new_max = 1.0

            from .models import Model

            class SimpleProbabilisticModel(Model):
                def __init__(self, model, new_min, new_max):
                    self.model = model
                    self.new_min = new_min
                    self.new_max = new_max
                    super(SimpleProbabilisticModel, self).__init__()

                def _log_density(self, stimulus):
                    smap = self.model.saliency_map(stimulus)
                    smap = new_min + smap / (new_max - new_min)
                    smap /= np.sum(smap)
                    return np.log(smap)

            return SimpleProbabilisticModel(model, new_min, new_max)

        prob_model = convert_model(self, minimum_value)
        if convert_gold_standard:
            prob_gold_standard = convert_model(gold_standard, minimum_value)
        else:
            prob_gold_standard = gold_standard

        return prob_model.kl_divergences(stimuli, prob_gold_standard)

    def image_based_kl_divergence(self, stimuli, gold_standard, minimum_value=1e-20, convert_gold_standard=True):
        """Calculate image-based KL-Divergences between model and gold standard averaged over stimuli

        for more details, see `image_based_kl_divergences`.
        """
        return np.mean(self.image_based_kl_divergences(stimuli, gold_standard,
                                                       minimum_value=minimum_value,
                                                       convert_gold_standard=convert_gold_standard))

    def CCs(self, stimuli, other, verbose=False):
        """ Calculate Correlation Coefficient Metric against some other model

        Returns performances for each stimulus. For performance over dataset,
        see `CC`
        """
        coeffs = []
        for s in tqdm(stimuli, disable=not verbose):
            smap1 = self.saliency_map(s).copy()
            smap1 -= smap1.mean()
            smap1 /= smap1.std()

            smap2 = other.saliency_map(s).copy()
            smap2 -= smap2.mean()
            smap2 /= smap2.std()

            coeffs.append(np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1])
        return np.asarray(coeffs)

    def CC(self, stimuli, other, verbose=False):
        return self.CCs(stimuli, other, verbose=verbose).mean()

    def NSSs(self, stimuli, fixations, verbose=False):
        values = np.empty(len(fixations.x))
        for n, s in enumerate(tqdm(stimuli, disable=not verbose)):
            smap = self.saliency_map(s).copy()
            mean = smap.mean()
            std = smap.std()

            inds = fixations.n == n

            _values = smap[fixations.y_int[inds], fixations.x_int[inds]]
            _values -= mean
            _values /= std
            values[inds] = _values

        return values

    def SIMs(self, stimuli, other, verbose=False):
        """ Calculate Similarity Metric against some other model

        Returns performances for each stimulus. For performance over dataset,
        see `SIM`
        """

        def _normalize_saliency_map(smap):
            if smap.min() < 0:
                smap = smap - smap.min()
            smap = smap / smap.sum()
            return smap

        values = []
        for s in tqdm(stimuli, disable=not verbose):
            smap1 = _normalize_saliency_map(self.saliency_map(s))

            smap2 = _normalize_saliency_map(other.saliency_map(s))

            values.append(np.min([smap1, smap2], axis=0).sum())
        return np.asarray(values)

    def SIM(self, stimuli, other, verbose=False):
        return self.SIMs(stimuli, other, verbose=verbose).mean()

    def __add__(self, other):
        if not isinstance(other, SaliencyMapModel):
            return NotImplemented

        return LambdaSaliencyMapModel([self, other], fn=lambda smaps: np.sum(smaps, axis=0, keepdims=False), caching=False)

    def __sub__(self, other):
        if not isinstance(other, SaliencyMapModel):
            return NotImplemented

        return LambdaSaliencyMapModel([self, other], fn=lambda smaps: smaps[0] - smaps[1], caching=False)

    def __mul__(self, other):
        if not isinstance(other, SaliencyMapModel):
            return NotImplemented

        return LambdaSaliencyMapModel([self, other], fn=lambda smaps: np.prod(smaps, axis=0, keepdims=False), caching=False)

    def __truediv__(self, other):
        if not isinstance(other, SaliencyMapModel):
            return NotImplemented

        return LambdaSaliencyMapModel([self, other], fn=lambda smaps: smaps[0] / smaps[1], caching=False)


class CachedSaliencyMapModel(SaliencyMapModel):
    """Saliency map model which uses only precached saliency maps
    """
    def __init__(self, cache_location, **kwargs):
        if cache_location is None:
            raise ValueError("CachedSaliencyMapModel needs a cache location!")
        super(CachedSaliencyMapModel, self).__init__(cache_location=cache_location, **kwargs)

    def _saliency_map(self, stimulus):
        raise NotImplementedError()


class MatlabSaliencyMapModel(SaliencyMapModel):
    """
    A model that creates it's saliency maps from a matlab script.

    The script has to take at least two arguments: The first argument
    will contain the filename which contains the stimulus (by default as png),
    the second argument contains the filename where the saliency map should be
    saved to (by default a .mat file). For more complicated scripts, you can
    overwrite the method `matlab_command`. It has to be a format string
    which takes the fields `stimulus` and `saliency_map` for the stimulus file
    and the saliency map file.
    """
    def __init__(self, script_file, stimulus_ext = '.png', saliency_map_ext='.mat', only_color_stimuli=False, **kwargs):
        """
        Initialize MatlabSaliencyModel

        Parameters
        ----------

        @type  script_file: string
        @param script_file: location of script file for Matlab/octave.
                            Matlab/octave will be run from this directory.

        @type  stimulus_ext: string, defaults to '.png'
        @param stimulus_ext: In which format the stimulus should be handed to the matlab script.

        @type  saliency_map_ext: string, defaults to '.png'
        @param saliency_map_ext: In which format the script will return the saliency map

        @type  only_color_stimuli: bool, defaults to `False`
        @param only_color_stimuli: If True, indicates that the script can handle only color stimuli.
                                   Grayscale stimuli will be converted to color stimuli by setting all
                                   RGB channels to the same value.
        """
        super(MatlabSaliencyMapModel, self).__init__(**kwargs)
        self.script_file = script_file
        self.stimulus_ext = stimulus_ext
        self.saliency_map_ext = saliency_map_ext
        self.only_color_stimuli = only_color_stimuli
        self.script_directory = os.path.dirname(script_file)
        script_name = os.path.basename(script_file)
        self.command, ext = os.path.splitext(script_name)

    def matlab_command(self, stimulus):
        """
        Construct the command to pass to matlab.

        Parameters
        ----------

        @type  stimulus: ndarray
        @param stimulus: The stimulus for which the saliency map should be generated.
                         In most cases, this argument should not be needed.

        @returns: string, the command to pass to matlab. The returned string has to be
                  a format string with placeholders for `stimulus` and `saliency_map`
                  where the files containing stimulus and saliency map will be inserted.
                  To change the type of these files, see the constructor.
        """
        return "{command}('{{stimulus}}', '{{saliency_map}}');".format(command=self.command)

    def _saliency_map(self, stimulus):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            stimulus_file = os.path.join(temp_dir, 'stimulus'+self.stimulus_ext)
            if self.only_color_stimuli:
                if stimulus.ndim == 2:
                    new_stimulus = np.empty((stimulus.shape[0], stimulus.shape[1], 3), dtype=stimulus.dtype)
                    for i in range(3):
                        new_stimulus[:, :, i] = stimulus
                    stimulus = new_stimulus
            if self.stimulus_ext == '.png':
                imsave(stimulus_file, stimulus)
            else:
                raise ValueError(self.stimulus_ext)

            saliency_map_file = os.path.join(temp_dir, 'saliency_map'+self.saliency_map_ext)

            command = self.matlab_command(stimulus).format(stimulus=stimulus_file,
                                                           saliency_map=saliency_map_file)

            run_matlab_cmd(command, cwd = self.script_directory)

            if self.saliency_map_ext == '.mat':
                saliency_map = loadmat(saliency_map_file)['saliency_map']
            else:
                raise ValueError(self.saliency_map_ext)

            return saliency_map


class FixationMap(SaliencyMapModel):
    """
    Fixation maps for given stimuli and fixations.

    With the keyword `kernel_size`, you can control whether
    the fixation map should be blured or just contain
    the actual fixations.
    """
    def __init__(self, stimuli, fixations, kernel_size=None, convolution_mode='reflect', *args, **kwargs):
        super(FixationMap, self).__init__(*args, **kwargs)

        self.xs = {}
        self.ys = {}
        for n in range(len(stimuli)):
            f = fixations[fixations.n == n]
            self.xs[stimuli.stimulus_ids[n]] = f.x.copy()
            self.ys[stimuli.stimulus_ids[n]] = f.y.copy()

        self.kernel_size = kernel_size
        self.convolution_mode = convolution_mode

    def _saliency_map(self, stimulus):
        stimulus = Stimulus(stimulus)
        stimulus_id = stimulus.stimulus_id
        if stimulus.stimulus_id not in self.xs:
            raise ValueError('No Fixations known for this stimulus!')
        saliency_map = np.zeros(stimulus.size)
        saliency_map[self.ys[stimulus_id].astype(int), self.xs[stimulus_id].astype(int)] = 1.0

        if self.kernel_size:
            saliency_map = gaussian_filter(saliency_map, self.kernel_size, mode=self.convolution_mode)
        return saliency_map


class ResizingSaliencyMapModel(SaliencyMapModel):
    def __init__(self, parent_model, verbose=True, **kwargs):
        if 'caching' not in kwargs:
            kwargs['caching'] = False
        super(ResizingSaliencyMapModel, self).__init__(**kwargs)
        self.parent_model = parent_model
        self.verbose = verbose

    def _saliency_map(self, stimulus):
        smap = self.parent_model.saliency_map(stimulus)

        target_shape = (stimulus.shape[0],
                        stimulus.shape[1])

        if smap.shape != target_shape:
            if self.verbose:
                print("Resizing saliency map", smap.shape, target_shape)
            x_factor = target_shape[1] / smap.shape[1]
            y_factor = target_shape[0] / smap.shape[0]

            smap = zoom(smap, [y_factor, x_factor], order=1, mode='nearest')

            assert smap.shape == target_shape

        return smap


class DisjointUnionSaliencyMapModel(GeneralSaliencyMapModel):
    #def __init__(self):
    #
    def _split_fixations(self, stimuli, fixations):
        """ return list of [(inds, model)]
        """
        raise NotImplementedError()

    def conditional_saliency_map(self, stimulus, *args, **kwargs):
        raise

    def eval_metric(self, metric_name, stimuli, fixations, **kwargs):
        result = np.empty(len(fixations.x))
        done = np.zeros_like(result).astype(bool)
        verbose = kwargs.get('verbose')
        for inds, model in tqdm(self._split_fixations(stimuli, fixations), disable = not verbose):
            assert done[inds].sum() == 0
            _f = fixations[inds]
            this_metric = getattr(model, metric_name)
            this_result = this_metric(stimuli, _f, **kwargs)
            result[inds] = this_result
            done[inds] = True
        assert all(done)
        return result

    def AUCs(self, stimuli, fixations, **kwargs):
        return self.eval_metric('AUCs', stimuli, fixations, **kwargs)

    def AUC(self, stimuli, fixations, **kwargs):
        if kwargs.get('nonfixations', 'uniform') == 'shuffled':
            kwargs = dict(kwargs)
            kwargs['nonfixations'] = FullShuffledNonfixationProvider(stimuli, fixations)
        return super(DisjointUnionSaliencyMapModel, self).AUC(stimuli, fixations, **kwargs)

    def NSSs(self, stimuli, fixations, **kwargs):
        return self.eval_metric('NSSs', stimuli, fixations, **kwargs)


class SubjectDependentSaliencyMapModel(DisjointUnionSaliencyMapModel):
    def __init__(self, subject_models, **kwargs):
        super(SubjectDependentSaliencyMapModel, self).__init__(**kwargs)
        self.subject_models = subject_models

    def _split_fixations(self, stimuli, fixations):
        for s in self.subject_models:
            yield fixations.subjects == s, self.subject_models[s]


class ExpSaliencyMapModel(SaliencyMapModel):
    def __init__(self, parent_model):
        super(ExpSaliencyMapModel, self).__init__(caching=False)
        self.parent_model = parent_model

    def _saliency_map(self, stimulus):
        return np.exp(self.parent_model.saliency_map(stimulus))


class BluringSaliencyMapModel(SaliencyMapModel):
    def __init__(self, parent_model, kernel_size, mode='nearest', **kwargs):
        super(BluringSaliencyMapModel, self).__init__(**kwargs)
        self.parent_model = parent_model
        self.kernel_size = kernel_size
        self.mode = mode

    def _saliency_map(self, stimulus):
        smap = self.parent_model.saliency_map(stimulus)
        smap = gaussian_filter(smap, self.kernel_size, mode=self.mode)
        return smap


class DigitizeMapModel(SaliencyMapModel):
    def __init__(self, parent_model, bins=256, return_ints=True):
        super(DigitizeMapModel, self).__init__(caching=False)
        self.parent_model = parent_model
        self.bins = bins
        self.return_ints = return_ints

    def _saliency_map(self, stimulus):
        smap = self.parent_model.saliency_map(stimulus)
        min = smap.min()
        max = smap.max()
        bins = np.linspace(min, max, num=self.bins+1)
        smap = np.digitize(smap, bins) - 1
        if self.return_ints:
            return smap
        else:
            return smap.astype(float)


class HistogramNormalizedSaliencyMapModel(SaliencyMapModel):
    def __init__(self, parent_model, histogram=None, **kwargs):
        super(HistogramNormalizedSaliencyMapModel, self).__init__(**kwargs)

        self.parent_model = parent_model

        if histogram is None:
            histogram = np.ones(256) / 256

        self.histogram = histogram
        self.histogram /= self.histogram.sum()
        self.bins = np.linspace(0, 1, len(self.histogram))
        self.cdf = np.cumsum(self.histogram)

    def _saliency_map(self, stimulus):
        smap = self.parent_model.saliency_map(stimulus)
        return normalize_saliency_map(smap, self.cdf, self.bins)


def export_model_to_hdf5(model, stimuli, filename, compression=9):
    assert isinstance(stimuli, FileStimuli)

    import h5py
    with h5py.File(filename, mode='w') as f:
        for k, s in enumerate(tqdm(stimuli)):
            stimulus_name = stimuli.filenames[k]
            _, stimulus_name = os.path.split(stimulus_name)

            smap = model.saliency_map(s)
            f.create_dataset(stimulus_name, data=smap, compression=compression)


class LambdaSaliencyMapModel(SaliencyMapModel):
    """Applies a function to a list of saliency maps from other models"""
    def __init__(self, parent_models, fn, **kwargs):
        super(LambdaSaliencyMapModel, self).__init__(**kwargs)
        self.parent_models = parent_models
        self.fn = fn

    def _saliency_map(self, stimulus):
        saliency_maps = [model.saliency_map(stimulus) for model in self.parent_models]
        return self.fn(saliency_maps)
