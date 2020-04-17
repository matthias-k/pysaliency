import numpy as np
import dill
import pytest

from pysaliency import GaussianSaliencyMapModel, Stimuli, Fixations
from pysaliency.saliency_map_conversion_theano import SaliencyMapConvertor, optimize_for_information_gain


@pytest.mark.theano
@pytest.mark.parametrize("optimize", [
    None,
    ['nonlinearity'],
    ['nonlinearity', 'centerbias'],
    ['nonlinearity', 'alpha', 'centerbias'],
    ['centerbias'],
    ['blur_radius'],
    ['blur_radius', 'nonlinearity']
])
def test_optimize_for_IG(optimize):
    # To speed up testing, we disable some optimizations
    import theano
    old_optimizer = theano.config.optimizer
    theano.config.optimizer = 'fast_compile'

    model = GaussianSaliencyMapModel()
    stimulus = np.random.randn(100, 100, 3)
    stimuli = Stimuli([stimulus])

    rst = np.random.RandomState(seed=42)
    N = 100000
    fixations = Fixations.create_without_history(
        x=rst.rand(N) * 100,
        y=rst.rand(N) * 100,
        n=np.zeros(N, dtype=int)
    )

    smc, res = optimize_for_information_gain(
        model,
        stimuli,
        fixations,
        optimize=optimize,
        blur_radius=3,
        verbose=2,
        maxiter=10,
        return_optimization_result=True)

    theano.config.optimizer = old_optimizer

    assert res.status in [
        0,  # success
        9,  # max iter reached
    ]

    assert smc


@pytest.mark.theano
def test_saliency_map_converter(tmpdir):
    import theano
    theano.config.floatX = 'float64'
    old_optimizer = theano.config.optimizer
    theano.config.optimizer = 'fast_compile'

    model = GaussianSaliencyMapModel()
    smc = SaliencyMapConvertor(model)
    smc.set_params(nonlinearity=np.ones(20),
                   centerbias=np.ones(12) * 2,
                   alpha=3,
                   blur_radius=4,
                   saliency_min=5,
                   saliency_max=6)

    theano.config.optimizer = old_optimizer

    pickle_file = tmpdir.join('object.pydat')
    with pickle_file.open(mode='wb') as f:
        dill.dump(smc, f)

    with pickle_file.open(mode='rb') as f:
        smc2 = dill.load(f)

    np.testing.assert_allclose(smc2.saliency_map_processing.nonlinearity_ys.get_value(), np.ones(20))
    np.testing.assert_allclose(smc2.saliency_map_processing.centerbias_ys.get_value(), np.ones(12) * 2)
    np.testing.assert_allclose(smc2.saliency_map_processing.alpha.get_value(), 3)
    np.testing.assert_allclose(smc2.saliency_map_processing.blur_radius.get_value(), 4)
    np.testing.assert_allclose(smc2.saliency_min, 5)
    np.testing.assert_allclose(smc2.saliency_max, 6)
