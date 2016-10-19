from __future__ import absolute_import, print_function, division, unicode_literals

import os
from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave
from scipy.ndimage import gaussian_filter

from tqdm import tqdm

from .generics import progressinfo
from .roc import general_roc

from .utils import TemporaryDirectory, run_matlab_cmd, Cache
from .datasets import Stimulus


def handle_stimulus(stimulus):
    """
    Make sure that a stimulus is a `Stimulus`-object
    """
    if not isinstance(stimulus, Stimulus):
        stimulus = Stimulus(stimulus)
    return stimulus


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
        if nonfixations not in ['uniform', 'shuffled']:
            nonfix_xs = []
            nonfix_ys = []
            for n in range(fixations.n.max()+1):
                inds = nonfixations.n == n
                nonfix_xs.append(nonfixations.x_int[inds].copy())
                nonfix_ys.append(nonfixations.y_int[inds].copy())

        if nonfixations == 'shuffled':
            nonfix_ys = []
            nonfix_xs = []
            widths = np.asarray([s[1] for s in stimuli.sizes]).astype(float)
            heights = np.asarray([s[0] for s in stimuli.sizes]).astype(float)
            for n in range(fixations.n.max()+1):
                inds = ~(fixations.n == n)
                xs = (fixations.x[inds].copy())
                ys = (fixations.y[inds].copy())

                other_ns = fixations.n[inds]
                xs *= stimuli.sizes[n][1]/widths[other_ns]
                ys *= stimuli.sizes[n][0]/heights[other_ns]

                nonfix_xs.append(xs.astype(int))
                nonfix_ys.append(ys.astype(int))
        for i in progressinfo(range(len(fixations.x)), verbose=verbose):
            out = self.conditional_saliency_map(stimuli.stimulus_objects[fixations.n[i]], fixations.x_hist[i], fixations.y_hist[i],
                                                fixations.t_hist[i], out=out)
            positives = np.asarray([out[fixations.y_int[i], fixations.x_int[i]]])
            if nonfixations == 'uniform':
                negatives = out.flatten()
            else:
                n = fixations.n[i]
                negatives = out[nonfix_ys[n], nonfix_xs[n]]
            this_roc, _, _ = general_roc(positives, negatives)
            rocs.setdefault(fixations.n[i], []).append(this_roc)
            rocs_per_fixation.append(this_roc)
#        if average is 'image':
#            rocs_per_image = [np.mean(rocs.get(n, [])) for n in range(fixations.n.max()+1)]
#            return rocs_per_image
#        else:
        return rocs_per_fixation

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

    def __init__(self, cache_location = None, caching=True):
        self._cache = Cache(cache_location)
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
        rocs = {}
        out = None
        if nonfixations not in ['uniform', 'shuffled']:
            nonfix_xs = []
            nonfix_ys = []
            for n in range(fixations.n.max()+1):
                inds = nonfixations.n == n
                nonfix_xs.append(nonfixations.x_int[inds].copy())
                nonfix_ys.append(nonfixations.y_int[inds].copy())

        if nonfixations == 'shuffled':
            nonfix_ys = []
            nonfix_xs = []
            widths = np.asarray([s[1] for s in stimuli.sizes]).astype(float)
            heights = np.asarray([s[0] for s in stimuli.sizes]).astype(float)

        for n in progressinfo(range(len(stimuli)), verbose=verbose):
            out = self.saliency_map(stimuli.stimulus_objects[n])
            inds = fixations.n == n
            positives = np.asarray(out[fixations.y_int[inds], fixations.x_int[inds]])
            if nonfixations == 'uniform':
                negatives = out.flatten()
            elif nonfixations == 'shuffled':
                inds = ~(fixations.n == n)
                xs = (fixations.x[inds].copy())
                ys = (fixations.y[inds].copy())

                other_ns = fixations.n[inds]
                xs *= stimuli.sizes[n][1]/widths[other_ns]
                ys *= stimuli.sizes[n][0]/heights[other_ns]
                xs = xs.astype(int)
                ys = ys.astype(int)
                negatives = out[ys, xs]
            else:
                negatives = out[nonfix_ys[n], nonfix_xs[n]]
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
        aucs = self.AUC_per_image(stimuli, fixations, nonfixations=nonfixations, verbose=verbose)
        if average == 'fixation':
            weights = np.zeros_like(aucs)
            for n in set(fixations.n):
                weights[n] = (fixations.n == n).mean()
            weights /= weights.sum()
            #indices, weights = np.unique(fixations.n, return_counts = True)
            #weights = weights[np.argsort(indices)].astype(float)
            #weights /= weights.sum()
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
        values = []
        for n, s in enumerate(tqdm(stimuli, disable=not verbose)):
            smap = self.saliency_map(s).copy()
            mean = smap.mean()
            std = smap.std()

            inds = fixations.n == n

            _values = smap[fixations.y_int[inds], fixations.x_int[inds]]
            _values -= mean
            _values /= std

            values.append(_values)
        return np.hstack(values)

    def NSS(self, stimuli, fixations, verbose=False):
        return self.NSSs(stimuli, fixations, verbose=verbose).mean()


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
    def __init__(self, stimuli, fixations, *args, **kwargs):
        kernel_size = kwargs.pop('kernel_size', None)
        super(FixationMap, self).__init__(*args, **kwargs)

        self.xs = {}
        self.ys = {}
        for n in range(len(stimuli)):
            f = fixations[fixations.n == n]
            self.xs[stimuli.stimulus_ids[n]] = f.x.copy()
            self.ys[stimuli.stimulus_ids[n]] = f.y.copy()

        self.kernel_size = kernel_size

    def _saliency_map(self, stimulus):
        stimulus = Stimulus(stimulus)
        stimulus_id = stimulus.stimulus_id
        if stimulus.stimulus_id not in self.xs:
            raise ValueError('No Fixations known for this stimulus!')
        saliency_map = np.zeros(stimulus.size)
        saliency_map[self.ys[stimulus_id].astype(int), self.xs[stimulus_id].astype(int)] = 1.0

        if self.kernel_size:
            saliency_map = gaussian_filter(saliency_map, self.kernel_size)
        return saliency_map
