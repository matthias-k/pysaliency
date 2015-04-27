import numpy as np
import dill

from nose.tools import assert_equal

from pysaliency import SaliencyMapConvertor, SaliencyMapModel

from test_helpers import TestWithData


class GaussianSaliencyMapModel(SaliencyMapModel):
    def _saliency_map(self, stimulus):
        height = stimulus.shape[0]
        width = stimulus.shape[1]
        YS, XS = np.mgrid[:height, :width]
        r_squared = (XS-0.5*width)**2 + (YS-0.5*height)**2
        size = np.sqrt(width**2+height**2)
        return np.ones((stimulus.shape[0], stimulus.shape[1]))*np.exp(-0.5*(r_squared/size))


class TestSaliencyMapConvertor(TestWithData):
    def test_pickle(self):
        import theano
        theano.config.floatX = 'float64'
        model = GaussianSaliencyMapModel()
        smc = SaliencyMapConvertor(model)
        smc.set_params(nonlinearity=np.ones(20),
                       centerbias = np.ones(12)*2,
                       alpha=3,
                       blur_radius=4,
                       saliency_min=5,
                       saliency_max=6)

        smc2 = self.pickle_and_reload(smc, pickler=dill)
        np.testing.assert_allclose(smc2.saliency_map_processing.nonlinearity_ys.get_value(), np.ones(20))
        np.testing.assert_allclose(smc2.saliency_map_processing.centerbias_ys.get_value(), np.ones(12)*2)
        np.testing.assert_allclose(smc2.saliency_map_processing.alpha.get_value(), 3)
        np.testing.assert_allclose(smc2.saliency_map_processing.blur_radius.get_value(), 4)
        np.testing.assert_allclose(smc2.saliency_min, 5)
        np.testing.assert_allclose(smc2.saliency_max, 6)
