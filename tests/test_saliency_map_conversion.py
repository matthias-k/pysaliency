import numpy as np
import dill

from pysaliency import SaliencyMapConvertor, SaliencyMapModel, Stimuli, Fixations, optimize_for_information_gain


class GaussianSaliencyMapModel(SaliencyMapModel):
    def _saliency_map(self, stimulus):
        height = stimulus.shape[0]
        width = stimulus.shape[1]
        YS, XS = np.mgrid[:height, :width]
        r_squared = (XS-0.5*width)**2 + (YS-0.5*height)**2
        size = np.sqrt(width**2+height**2)
        return np.ones((stimulus.shape[0], stimulus.shape[1]))*np.exp(-0.5*(r_squared/size))


def test_optimize_for_IG():
    model = GaussianSaliencyMapModel()
    stimulus = np.random.randn(100, 100, 3)
    stimuli = Stimuli([stimulus])

    rst = np.random.RandomState(seed=42)
    N = 100000
    fixations = Fixations.create_without_history(
        x = rst.rand(N)*100,
        y = rst.rand(N)*100,
        n = np.zeros(N, dtype=int)
    )

    smc = optimize_for_information_gain(
        model,
        stimuli,
        fixations,
        blur_radius=3,
        verbose=2,
        maxiter=10)

    assert smc


def test_saliency_map_converter(tmpdir):
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

    pickle_file = tmpdir.join('object.pydat')
    with pickle_file.open(mode='wb') as f:
        dill.dump(smc, f)

    with pickle_file.open(mode='rb') as f:
        smc2 = dill.load(f)

    np.testing.assert_allclose(smc2.saliency_map_processing.nonlinearity_ys.get_value(), np.ones(20))
    np.testing.assert_allclose(smc2.saliency_map_processing.centerbias_ys.get_value(), np.ones(12)*2)
    np.testing.assert_allclose(smc2.saliency_map_processing.alpha.get_value(), 3)
    np.testing.assert_allclose(smc2.saliency_map_processing.blur_radius.get_value(), 4)
    np.testing.assert_allclose(smc2.saliency_min, 5)
    np.testing.assert_allclose(smc2.saliency_max, 6)
