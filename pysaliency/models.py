from __future__ import absolute_import, print_function, division, unicode_literals

import os

import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave

import generics
from .roc import general_roc

from .utils import TemporaryDirectory, run_matlab_cmd
from .datasets import Stimulus


def handle_stimulus(stimulus):
    """
    Make sure that a stimulus is a `Stimulus`-object
    """
    if not isinstance(stimulus, Stimulus):
        stimulus = Stimulus(stimulus)
    return stimulus


class GeneralSaliencyMapModel(object):
    """
    Most general saliency model class. The model is neither
    assumed to be time-independet nor to be a probabilistic
    model.
    """

    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist, out=None):
        """
        Return the models saliency map prediction depending on a fixation history
        for the n-th image.
        """
        raise NotImplementedError()

    def AUCs(self, stimuli, fixations, nonfixations='uniform'):
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
        for i in generics.progressinfo(range(len(fixations.x))):
            out = self.conditional_saliency_map(stimuli.stimulus_objects[fixations.n[i]], fixations.x_hist[i], fixations.y_hist[i],
                                                fixations.t_hist[i], out=out)
            positives = np.asarray([out[fixations.y[i], fixations.x[i]]])
            if nonfixations == 'uniform':
                negatives = out.flatten()
            elif nonfixations == 'shuffled':
                n = fixations.n[i]
                negatives = out[nonfix_ys[n], nonfix_xs[n]]
            else:
                negatives = out[nonfix_ys[n], nonfix_xs[n]]
            this_roc, _, _ = general_roc(positives, negatives)
            rocs.setdefault(fixations.n[i], []).append(this_roc)
            rocs_per_fixation.append(this_roc)
#        if average is 'image':
#            rocs_per_image = [np.mean(rocs.get(n, [])) for n in range(fixations.n.max()+1)]
#            return rocs_per_image
#        else:
        return rocs_per_fixation

    def AUC(self, stimuli, fixations, nonfixations='uniform', average='fixation'):
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
        aucs = self.AUCs(stimuli, fixations, nonfixations=nonfixations)
        return np.mean(aucs)

    def fixation_based_KL_divergence(self, stimuli, fixations, nonfixations='shuffled', bins=10):
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

        :rtype : ndarray
        :return : list of AUC scores for each fixation,
                  ordered as in `fixations.x` (average=='fixation' or None)
                  or by image numbers (average=='image')
        """



class SaliencyMapModel(GeneralSaliencyMapModel):
    """
    Most model class for saliency maps. The model is assumed
    to be stationary in time (i.e. all fixations are independent)
    but the model is not explicitly a probabilistic model.
    """

    def __init__(self):
        self._saliency_map_cache = {}

    def saliency_map(self, stimulus):
        """
        Get saliency map for given stimulus.

        To overwrite this function, overwrite `_saliency_map` as otherwise
        the caching mechanism is disabled.
        """
        stimulus = handle_stimulus(stimulus)
        stimulus_id = stimulus.stimulus_id
        if not stimulus_id in self._saliency_map_cache:
            self._saliency_map_cache[stimulus_id] = self._saliency_map(stimulus.stimulus_data)
        return self._saliency_map_cache[stimulus_id]

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
    def __init__(self, script_file, stimulus_ext = '.png', saliency_map_ext='.mat', only_color_stimuli=False):
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
        super(MatlabSaliencyMapModel, self).__init__()
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
