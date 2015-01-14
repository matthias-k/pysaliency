import numpy as np
import dill

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
        model = GaussianSaliencyMapModel()
        smc = SaliencyMapConvertor(model)

        smc2 = self.pickle_and_reload(smc, pickler=dill)
