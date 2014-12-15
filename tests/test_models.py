from __future__ import absolute_import, division, print_function, unicode_literals

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


class TestAUC(object):
    def setUp(self):
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
        self.f = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)

    def test_constant(self):

        stimuli = pysaliency.Stimuli([np.random.randn(600, 1000, 3),
                                      np.random.randn(600, 1000, 3)])
        csmm = ConstantSaliencyModel()

        aucs = csmm.AUCs(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(aucs, np.ones(len(self.f.x))*0.5)

        aucs = csmm.AUCs(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(aucs, np.ones(len(self.f.x))*0.5)

    def test_gauss(self):

        stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                      np.random.randn(40, 40, 3)])
        gsmm = GaussianSaliencyModel()

        aucs = gsmm.AUCs(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(aucs, [0.099375,  0.158125,  0.241875,  0.241875,
                                          0.241875,  0.291875, 0.509375,  0.138125],
                                   rtol=1e-6)

        aucs = gsmm.AUCs(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(aucs, [0.0,         0.33333333,  0.33333333,  0.33333333,
                                          0.33333333,  1.,          1.,          0.2],
                                   rtol=1e-6)


class TestFixationBasedKLDivergence(object):
    def setUp(self):
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
        self.f = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)

    def test_constant(self):
        stimuli = pysaliency.Stimuli([np.random.randn(600, 1000, 3),
                                      np.random.randn(600, 1000, 3)])
        csmm = ConstantSaliencyModel()

        fb_kl = csmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(fb_kl, 0.0)

        fb_kl = csmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(fb_kl, 0.0)

    def test_gauss(self):
        stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                      np.random.randn(40, 40, 3)])
        gsmm = GaussianSaliencyModel()

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(fb_kl, 0.8763042309253437)

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(fb_kl, 0.0)


class TestLogLikelihoods(object):
    def setUp(self):
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
        self.f = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)

    def test_constant(self):
        stimuli = pysaliency.Stimuli([np.random.randn(600, 1000, 3),
                                      np.random.randn(600, 1000, 3)])
        csmm = ConstantSaliencyModel()

        log_likelihoods = csmm.log_likelihoods(stimuli, self.f)
        np.testing.assert_allclose(log_likelihoods, -np.log(600*1000))

    def test_gauss(self):
        stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                      np.random.randn(40, 40, 3)])
        gsmm = GaussianSaliencyModel()

        log_likelihoods = gsmm.log_likelihoods(stimuli, self.f)
        np.testing.assert_allclose(log_likelihoods, np.array([-10.276835,  -9.764182,  -9.286885,  -9.286885,
                                                              -9.286885,   -9.057075,  -8.067126,  -9.905604]))
