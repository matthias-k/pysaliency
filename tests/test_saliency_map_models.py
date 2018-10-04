from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

import pysaliency
import pysaliency.saliency_map_models


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


@pytest.fixture
def fixations():
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


def test_auc_constant(fixations):
    stimuli = pysaliency.Stimuli([np.random.randn(600, 1000, 3),
                                  np.random.randn(600, 1000, 3)])
    csmm = ConstantSaliencyMapModel()

    aucs = csmm.AUCs(stimuli, fixations, nonfixations='uniform')
    np.testing.assert_allclose(aucs, np.ones(len(fixations.x))*0.5)

    aucs = csmm.AUCs(stimuli, fixations, nonfixations='shuffled')
    np.testing.assert_allclose(aucs, np.ones(len(fixations.x))*0.5)

    auc = csmm.AUC(stimuli, fixations, nonfixations=fixations)
    np.testing.assert_allclose(auc, 0.5)


def test_auc_gauss(fixations):
    stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                  np.random.randn(40, 40, 3)])
    gsmm = GaussianSaliencyMapModel()

    aucs = gsmm.AUCs(stimuli, fixations, nonfixations='uniform')
    np.testing.assert_allclose(aucs, [0.099375,  0.158125,  0.241875,  0.241875,
                                      0.241875,  0.291875, 0.509375,  0.138125],
                               rtol=1e-6)

    aucs = gsmm.AUCs(stimuli, fixations, nonfixations='shuffled')
    np.testing.assert_allclose(aucs, [0.0,         0.33333333,  0.33333333,  0.33333333,
                                      0.33333333,  1.,          1.,          0.2],
                               rtol=1e-6)

    auc = gsmm.AUC(stimuli, fixations, nonfixations=fixations)
    np.testing.assert_allclose(auc, 0.5)


    aucs_single = pysaliency.GeneralSaliencyMapModel.AUCs(gsmm, stimuli, fixations)
    aucs_combined = gsmm.AUCs(stimuli, fixations, nonfixations='uniform')
    np.testing.assert_allclose(aucs_single, aucs_combined)


def test_fixation_based_kldiv_constant(fixations):
    stimuli = pysaliency.Stimuli([np.random.randn(600, 1000, 3),
                                  np.random.randn(600, 1000, 3)])
    csmm = ConstantSaliencyMapModel()

    fb_kl = csmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations='uniform')
    np.testing.assert_allclose(fb_kl, 0.0)

    fb_kl = csmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations='shuffled')
    np.testing.assert_allclose(fb_kl, 0.0)

    fb_kl = csmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations=fixations)
    np.testing.assert_allclose(fb_kl, 0.0)


def test_fixation_based_kldiv_gauss(fixations):
    stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                  np.random.randn(40, 40, 3)])
    gsmm = GaussianSaliencyMapModel()

    fb_kl = gsmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations='uniform')
    np.testing.assert_allclose(fb_kl, 0.4787711930295902)

    fb_kl = gsmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations='shuffled')
    np.testing.assert_allclose(fb_kl, 0.0)

    fb_kl = gsmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations=fixations)
    np.testing.assert_allclose(fb_kl, 0.0)


def test_fixation_based_kldiv_mixed(fixations):
    stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                  np.random.randn(40, 40, 3)])
    gsmm = MixedSaliencyMapModel()

    fb_kl = gsmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations='uniform')
    np.testing.assert_allclose(fb_kl, 0.19700844437943388)

    fb_kl = gsmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations='shuffled')
    np.testing.assert_allclose(fb_kl, 5.874655219107867)

    fb_kl = gsmm.fixation_based_KL_divergence(stimuli, fixations, nonfixations=fixations)
    np.testing.assert_allclose(fb_kl, 0.0)


def test_image_based_kldiv_gauss(fixations):
    stimuli = pysaliency.Stimuli([np.random.randn(40, 40, 3),
                                  np.random.randn(40, 40, 3)])
    gsmm = GaussianSaliencyMapModel()
    constant_gold = ConstantSaliencyMapModel()
    gold = pysaliency.FixationMap(stimuli, fixations, kernel_size = 10)

    ib_kl = gsmm.image_based_kl_divergence(stimuli, gsmm)
    np.testing.assert_allclose(ib_kl, 0.0)

    ib_kl = gsmm.image_based_kl_divergence(stimuli, constant_gold)
    np.testing.assert_allclose(ib_kl, 0.8152272380729648)

    ib_kl = gsmm.image_based_kl_divergence(stimuli, gold)
    np.testing.assert_allclose(ib_kl, 1.961124862592289)

    ib_kl = gold.image_based_kl_divergence(stimuli, gold)
    np.testing.assert_allclose(ib_kl, 0.0)


def test_full_shuffled_nonfixation_provider():
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
    fixations = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)
    stimuli = pysaliency.Stimuli([np.random.randn(50, 50, 3),
                                  np.random.randn(50, 50, 3),
                                  np.random.randn(100, 200, 3)])

    from pysaliency.saliency_map_models import FullShuffledNonfixationProvider
    prov = FullShuffledNonfixationProvider(stimuli, fixations)


    xs, ys = prov(stimuli, fixations, 0)
    assert len(xs) == (fixations.n != 0).sum()
    np.testing.assert_allclose(xs, [1, 10, 3, 1, 1, 8, 1])
    np.testing.assert_allclose(ys, [21, 25, 33, 20, 21, 21, 22])


def test_lambda_saliency_map_model():
    stimuli = pysaliency.Stimuli([np.random.randn(50, 50, 3),
                                  np.random.randn(50, 50, 3),
                                  np.random.randn(100, 200, 3)])
    m1 = ConstantSaliencyMapModel()
    m2 = GaussianSaliencyMapModel()
    fn1 = lambda smaps: np.exp(smaps[0])
    fn2 = lambda smaps: np.sum(smaps, axis=0)
    lambda_model_1 = pysaliency.saliency_map_models.LambdaSaliencyMapModel([m1, m2], fn1)
    lambda_model_2 = pysaliency.saliency_map_models.LambdaSaliencyMapModel([m1, m2], fn2)

    for s in stimuli:
        smap1 = m1.saliency_map(s)
        smap2 = m2.saliency_map(s)
        np.testing.assert_allclose(lambda_model_1.saliency_map(s), fn1([smap1, smap2]))
        np.testing.assert_allclose(lambda_model_2.saliency_map(s), fn2([smap1, smap2]))


def test_saliency_map_model_operators():
    stimuli = pysaliency.Stimuli([np.random.randn(50, 50, 3),
                                  np.random.randn(50, 50, 3),
                                  np.random.randn(100, 200, 3)])
    m1 = ConstantSaliencyMapModel()
    m2 = GaussianSaliencyMapModel()

    for s in stimuli:
        smap1 = m1.saliency_map(s)
        smap2 = m2.saliency_map(s)
        np.testing.assert_allclose((m1+m2).saliency_map(s), smap1 + smap2)
        np.testing.assert_allclose((m1-m2).saliency_map(s), smap1 - smap2)
        np.testing.assert_allclose((m1*m2).saliency_map(s), smap1 * smap2)
        np.testing.assert_allclose((m1/m2).saliency_map(s), smap1 / smap2)
