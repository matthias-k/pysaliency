from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

import pysaliency


class ConstantSaliencyModel(pysaliency.Model):
    def _log_density(self, stimulus):
        return np.zeros((stimulus.shape[0], stimulus.shape[1])) - np.log(stimulus.shape[0]) - np.log(stimulus.shape[1])


class GaussianSaliencyModel(pysaliency.Model):
    def _log_density(self, stimulus):
        height = stimulus.shape[0]
        width = stimulus.shape[1]
        YS, XS = np.mgrid[:height, :width]
        r_squared = (XS-0.5*width)**2 + (YS-0.5*height)**2
        size = np.sqrt(width**2+height**2)
        values = np.ones((stimulus.shape[0], stimulus.shape[1]))*np.exp(-0.5*(r_squared/size))
        density = values / values.sum()
        return np.log(density)


@pytest.fixture
def fixation_trains():
    xs_trains = [
        [0, 1, 2],
        [2, 2],
        [1, 5, 3]]
    ys_trains = [
        [10, 11, 12],
        [12, 12],
        [21, 25, 33]]
    ts_trains = [
        [0, 200, 600],
        [100, 400],
        [50, 500, 900]]
    ns = [0, 0, 1]
    subjects = [0, 1, 1]
    return pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)


@pytest.fixture
def stimuli():
    return pysaliency.Stimuli([np.random.randn(40, 40, 3),
                               np.random.randn(40, 40, 3)])


def test_log_likelihood_constant(stimuli, fixation_trains):
    csmm = ConstantSaliencyModel()

    log_likelihoods = csmm.log_likelihoods(stimuli, fixation_trains)
    np.testing.assert_allclose(log_likelihoods, -np.log(40*40))


def test_log_likelihood_gauss(stimuli, fixation_trains):
    gsmm = GaussianSaliencyModel()

    log_likelihoods = gsmm.log_likelihoods(stimuli, fixation_trains)
    np.testing.assert_allclose(log_likelihoods, np.array([-10.276835,  -9.764182,  -9.286885,  -9.286885,
                                                          -9.286885,   -9.057075,  -8.067126,  -9.905604]))
    log_likelihoods = pysaliency.ScanpathModel.log_likelihoods(gsmm, stimuli, fixation_trains)
    np.testing.assert_allclose(log_likelihoods, np.array([-10.276835,  -9.764182,  -9.286885,  -9.286885,
                                                          -9.286885,   -9.057075,  -8.067126,  -9.905604]))


# @pytest.mark.parametrize("library", ['tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("library", ['torch', 'numpy'])
def test_shuffled_baseline_model(stimuli, library):
    # TODO: implement actual test
    model = GaussianSaliencyModel()
    shuffled_model = pysaliency.models.ShuffledBaselineModel(model, stimuli, library=library)

    assert model.log_density(stimuli[0]).shape == shuffled_model.log_density(stimuli[0]).shape


def test_sampling(stimuli):
    model = GaussianSaliencyModel()
    fixations = model.sample(stimuli, train_counts=10, lengths=3)
    assert len(fixations.train_xs) == len(stimuli) * 10
    assert len(fixations.x) == len(stimuli) * 10 * 3


@pytest.fixture
def long_stimuli():
    return pysaliency.Stimuli([np.random.randn(40, 60, 3) for index in range(1000)])


@pytest.fixture
def test_model(long_stimuli):
    class TestModel(pysaliency.Model):
        def __init__(self, stimuli, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stimuli = stimuli

        def _log_density(self, stimulus):
            stimulus = pysaliency.datasets.as_stimulus(stimulus)
            stimulus_index = self.stimuli.stimulus_ids.index(stimulus.stimulus_id)
            relative_index = stimulus_index / len(self.stimuli)

            this_model = pysaliency.models.GaussianModel(center_x=relative_index, center_y=relative_index)

            return this_model.log_density(stimulus)

    return TestModel(long_stimuli)


@pytest.fixture
def pixel_model(long_stimuli):
    class TestModel(pysaliency.Model):
        def __init__(self, stimuli, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stimuli = stimuli

        def _log_density(self, stimulus):
            stimulus = pysaliency.datasets.as_stimulus(stimulus)
            stimulus_index = self.stimuli.stimulus_ids.index(stimulus.stimulus_id)

            density = np.zeros(stimulus.size)
            density[stimulus_index, stimulus_index] = 1

            return np.log(density)

    return TestModel(long_stimuli[:40])


def test_average_predictions(long_stimuli, pixel_model):
    def log_density_iter(model, stimuli):
        return (model.log_density(s) for s in stimuli[:40])

    average_log_density = pysaliency.models.average_predictions(list(log_density_iter(pixel_model, long_stimuli)), library='torch')
    average_density = np.exp(average_log_density)
    np.testing.assert_allclose(np.diag(average_density), 1/40)


def test_average_predictions_iter(long_stimuli, test_model):
    def log_density_iter(model, stimuli):
        return (model.log_density(s) for s in stimuli)

    average_log_density_iter = pysaliency.models.average_predictions(log_density_iter(test_model, long_stimuli), library='torch')
    average_log_density_list = pysaliency.models.average_predictions(list(log_density_iter(test_model, long_stimuli)), library='torch')

    np.testing.assert_allclose(average_log_density_iter, average_log_density_list)


def test_average_predictions_iter(long_stimuli, test_model):
    def log_density_iter(model, stimuli):
        return (model.log_density(s) for s in stimuli)

    average_log_density_iter = pysaliency.models.average_predictions(
        log_density_iter(test_model, long_stimuli), library='numpy',
        log_density_count=len(long_stimuli),
        maximal_chunk_size=10,
        verbose=True,
    )
    average_log_density_list = pysaliency.models.average_predictions(list(log_density_iter(test_model, long_stimuli)), library='numpy')

    np.testing.assert_allclose(average_log_density_iter, average_log_density_list, rtol=1e-6)



def test_average_predictions_torch(long_stimuli, test_model):
    log_densities = [test_model.log_density(s) for s in long_stimuli[:20]]

    average_log_density_torch = pysaliency.models.average_predictions(log_densities, library='torch')
    average_log_density_numpy = pysaliency.models.average_predictions(log_densities, library='numpy')

    np.testing.assert_allclose(average_log_density_torch, average_log_density_numpy)


@pytest.mark.skip("need to fix tensorflow, convert to tf2")
def test_average_predictions_tensorflow(long_stimuli, test_model):
    log_densities = [test_model.log_density(s) for s in long_stimuli[:20]]

    average_log_density_tf = pysaliency.models.average_predictions(log_densities, library='tensorflow')
    average_log_density_numpy = pysaliency.models.average_predictions(log_densities, library='numpy')

    np.testing.assert_allclose(average_log_density_tf, average_log_density_numpy)