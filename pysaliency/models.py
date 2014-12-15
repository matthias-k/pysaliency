from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod

import numpy as np

import generics
from .saliency_map_models import GeneralSaliencyMapModel, SaliencyMapModel, handle_stimulus


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


class Model(GeneralModel, SaliencyMapModel):
    """
    Time independend probabilistic saliency model.

    Inheriting classes have to implement `_log_density`.
    """
    def __init__(self):
        super(Model, self).__init__()
        self._log_density_cache = {}

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
