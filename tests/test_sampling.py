import numpy as np

from pysaliency.saliency_map_models import GaussianSaliencyMapModel
from pysaliency.sampling_models import SamplingModelMixin, ScanpathSamplingModelMixin


def test_fixation_sampling():
    class SamplingModel(SamplingModelMixin, GaussianSaliencyMapModel):
        def sample_fixation(self, stimulus, x_hist, y_hist, t_hist, attributes=None, verbose=False, rst=None):
            return x_hist[-1] + 1, y_hist[-1] + 1, t_hist[-1] + 1

    model = SamplingModel()

    xs, ys, ts = model.sample_scanpath(np.zeros((40, 40, 3)), [0], [1], [2], 4)
    assert xs == [0, 1, 2, 3, 4]
    assert ys == [1, 2, 3, 4, 5]
    assert ts == [2, 3, 4, 5, 6]


def test_scanpath_sampling():
    class SamplingModel(ScanpathSamplingModelMixin, GaussianSaliencyMapModel):
        def sample_scanpath(self, stimulus, x_hist, y_hist, t_hist, samples, attributes=None, verbose=False, rst=None):
            return (
                list(x_hist) + [x_hist[-1]] * samples,
                list(y_hist) + [y_hist[-1]] * samples,
                list(t_hist) + [t_hist[-1]] * samples
            )

    model = SamplingModel()

    x, y, t = model.sample_fixation(np.zeros((40, 40, 3)), [0], [1], [2])
    assert x == 0
    assert y == 1
    assert t == 2
