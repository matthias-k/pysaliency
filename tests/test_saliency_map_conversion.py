import numpy as np
import pytest

import pysaliency
from pysaliency import optimize_for_information_gain
from pysaliency.models import SaliencyMapNormalizingModel


@pytest.fixture
def stimuli():
    return pysaliency.Stimuli([np.random.randint(0, 255, size=(25, 30, 3)) for i in range(50)])


@pytest.fixture
def saliency_model():
    return pysaliency.GaussianSaliencyMapModel(center_x=0.15, center_y=0.85, width=0.2)


@pytest.fixture
def transformed_saliency_model(saliency_model):
    return pysaliency.saliency_map_models.LambdaSaliencyMapModel(
        [saliency_model],
        fn=lambda smaps: np.sqrt(smaps[0]),
    )


@pytest.fixture
def probabilistic_model(saliency_model):
    blurred_model = pysaliency.BluringSaliencyMapModel(saliency_model, kernel_size=5.0)
    centerbias_model = pysaliency.saliency_map_models.LambdaSaliencyMapModel(
        [pysaliency.GaussianSaliencyMapModel(width=0.5)],
        fn=lambda smaps: 1.0 * smaps[0],
    )
    model_with_centerbias = blurred_model * centerbias_model
    probabilistic_model = SaliencyMapNormalizingModel(model_with_centerbias)

    return probabilistic_model


@pytest.fixture
def fixations(stimuli, probabilistic_model):
    return probabilistic_model.sample(stimuli, 1000, rst=np.random.RandomState(seed=42))


@pytest.fixture(params=["torch", "theano"])
def framework(request):

    if request.param == 'theano':
        import theano
        old_optimizer = theano.config.optimizer
        theano.config.optimizer = 'fast_compile'

    yield request.param

    if request.param == 'theano':
        theano.config.optimize = old_optimizer


def test_optimize_for_information_gain(stimuli, fixations, transformed_saliency_model, probabilistic_model, framework):
    expected_information_gain = probabilistic_model.information_gain(stimuli, fixations, average='image')

    model1, ret1 = optimize_for_information_gain(
        transformed_saliency_model,
        stimuli,
        fixations,
        average='fixations',
        verbose=2,
        batch_size=1 if framework == 'theano' else 10,
        minimize_options={'verbose': 10} if framework == 'torch' else None,
        maxiter=50,
        blur_radius=2.0,
        return_optimization_result=True,
        framework=framework,
    )

    reached_information_gain = model1.information_gain(stimuli, fixations, average='image')

    print(expected_information_gain, reached_information_gain)
    assert reached_information_gain >= expected_information_gain - 0.01
