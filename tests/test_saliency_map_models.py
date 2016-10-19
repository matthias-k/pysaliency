from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pysaliency


class ConstantSaliencyMapModel(pysaliency.SaliencyMapModel):
    def _saliency_map(self, stimulus):
        return np.ones((stimulus.shape[0], stimulus.shape[1]))


class GaussianSaliencyMapModel(pysaliency.SaliencyMapModel):
    def _saliency_map(self, stimulus):
        height = stimulus.shape[0]
        width = stimulus.shape[1]
        YS, XS = np.mgrid[:height, :width]
        r_squared = (XS-0.5*width)**2 + (YS-0.5*height)**2
        size = np.sqrt(width**2+height**2)
        return np.ones((stimulus.shape[0], stimulus.shape[1]))*np.exp(-0.5*(r_squared/size))


class MixedSaliencyMapModel(pysaliency.SaliencyMapModel):
    def __init__(self, *args, **kwargs):
        super(MixedSaliencyMapModel, self).__init__(*args, **kwargs)
        self.count = 0
        self.constant_model = ConstantSaliencyMapModel()
        self.gaussian_model = GaussianSaliencyMapModel()

    def _saliency_map(self, stimulus):
        self.count += 1
        if self.count % 2 == 1:
            return self.constant_model.saliency_map(stimulus)
        else:
            return self.gaussian_model.saliency_map(stimulus)


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
        csmm = ConstantSaliencyMapModel()

        aucs = csmm.AUCs(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(aucs, np.ones(len(self.f.x))*0.5)

        aucs = csmm.AUCs(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(aucs, np.ones(len(self.f.x))*0.5)

        auc = csmm.AUC(stimuli, self.f, nonfixations=self.f)
        np.testing.assert_allclose(auc, 0.5)

    def test_gauss(self):
        stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                      np.random.randn(40, 40, 3)])
        gsmm = GaussianSaliencyMapModel()

        aucs = gsmm.AUCs(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(aucs, [0.099375,  0.158125,  0.241875,  0.241875,
                                          0.241875,  0.291875, 0.509375,  0.138125],
                                   rtol=1e-6)

        aucs = gsmm.AUCs(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(aucs, [0.0,         0.33333333,  0.33333333,  0.33333333,
                                          0.33333333,  1.,          1.,          0.2],
                                   rtol=1e-6)

        auc = gsmm.AUC(stimuli, self.f, nonfixations=self.f)
        np.testing.assert_allclose(auc, 0.5)


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
        csmm = ConstantSaliencyMapModel()

        fb_kl = csmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(fb_kl, 0.0)

        fb_kl = csmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(fb_kl, 0.0)

        fb_kl = csmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations=self.f)
        np.testing.assert_allclose(fb_kl, 0.0)

    def test_gauss(self):
        stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                      np.random.randn(40, 40, 3)])
        gsmm = GaussianSaliencyMapModel()

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(fb_kl, 0.4787711930295902)

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(fb_kl, 0.0)

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations=self.f)
        np.testing.assert_allclose(fb_kl, 0.0)

    def test_mixed(self):
        stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                      np.random.randn(40, 40, 3)])
        gsmm = MixedSaliencyMapModel()

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='uniform')
        np.testing.assert_allclose(fb_kl, 0.19700844437943388)

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations='shuffled')
        np.testing.assert_allclose(fb_kl, 5.874655219107867)

        fb_kl = gsmm.fixation_based_KL_divergence(stimuli, self.f, nonfixations=self.f)
        np.testing.assert_allclose(fb_kl, 0.0)


class TestImageBasedKLDivergence(object):
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

    def test_gauss(self):
        stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                      np.random.randn(40, 40, 3)])
        gsmm = GaussianSaliencyMapModel()
        constant_gold = ConstantSaliencyMapModel()
        gold = pysaliency.FixationMap(stimuli, self.f, kernel_size = 10)

        ib_kl = gsmm.image_based_kl_divergence(stimuli, gsmm)
        np.testing.assert_allclose(ib_kl, 0.0)

        ib_kl = gsmm.image_based_kl_divergence(stimuli, constant_gold)
        np.testing.assert_allclose(ib_kl, 0.8152272380729648)

        ib_kl = gsmm.image_based_kl_divergence(stimuli, gold)
        np.testing.assert_allclose(ib_kl, 1.961124862592289)

        ib_kl = gold.image_based_kl_divergence(stimuli, gold)
        np.testing.assert_allclose(ib_kl, 0.0)


class TestFullShuffledNonfixationProvider(object):
    def setUp(self):
        xs_trains = [
            [0, 1, 2],
            [2, 2],
            [1, 10, 3],
            [4, 5, 33, 7]]
        ys_trains = [
            [10, 11, 12],
            [12, 12],
            [21, 25, 33],
            [41, 42, 43, 44]]
        ts_trains = [
            [0, 200, 600],
            [100, 400],
            [50, 500, 900],
            [0, 1, 2, 3]]
        ns = [0, 0, 1, 2]
        subjects = [0, 1, 1, 0]
        self.f = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)
        self.stimuli = pysaliency.Stimuli([np.random.randn(50, 50, 3),
                                           np.random.randn(50, 50, 3),
                                           np.random.randn(100, 200, 3)])

    def test_shuffled(self):
        from pysaliency.saliency_map_models import FullShuffledNonfixationProvider
        prov = FullShuffledNonfixationProvider(self.stimuli, self.f)


        xs, ys = prov(self.stimuli, self.f, 0)
        assert len(xs) == (self.f.n != 0).sum()
        np.testing.assert_allclose(xs, [1, 10, 3, 1, 1, 8, 1])
        np.testing.assert_allclose(ys, [21, 25, 33, 20, 21, 21, 22])

