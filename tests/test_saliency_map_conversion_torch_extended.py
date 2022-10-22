import numpy as np
import pytest

import pysaliency
from pysaliency.saliency_map_conversion import optimize_for_information_gain
from pysaliency.saliency_map_conversion_torch import SaliencyMapProcessing, SaliencyMapProcessingModel
from pysaliency import Stimuli, Fixations, GaussianSaliencyMapModel

import torch


@pytest.fixture
def stimuli():
    return pysaliency.Stimuli([np.random.randint(0, 255, size=(25, 30, 3)) for i in range(50)])


@pytest.fixture
def saliency_model():
    return pysaliency.GaussianSaliencyMapModel(center_x=0.15, center_y=0.85, width=0.5)


#@pytest.fixture(params=[0, 1, 2, 3, 4, 12])
@pytest.fixture(params=[
    0,
    #1,
    2,
    18,
])
def num_nonlinearity(request):
    return request.param

@pytest.fixture(params=[
    False,
    True
])
def is_blurring(request):
    return request.param


@pytest.fixture(params=[
    0,
    #1,
    2,
    14,
])
def num_centerbias(request):
    return request.param


@pytest.fixture(params=[
    False,
    True
])
def has_alpha(request):
    return request.param


@pytest.fixture
def probabilistic_model(saliency_model, is_blurring, num_nonlinearity, has_alpha, num_centerbias):
    saliency_map_processing = SaliencyMapProcessing(
        nonlinearity_values='logdensity',
        num_nonlinearity=num_nonlinearity,
        num_centerbias=num_centerbias,
        blur_radius=3 if is_blurring else 0,
    )

    with torch.no_grad():
        if num_nonlinearity > 0:
            # set nonlinearity
            #print("OLD", saliency_map_processing.nonlinearity.ys)
            old_exp_sum = torch.exp(saliency_map_processing.nonlinearity.ys).sum().detach().cpu().numpy()
            new_ys = 7 * np.linspace(0, 1, num_nonlinearity)**2
            new_ys -= np.log(old_exp_sum)
            saliency_map_processing.nonlinearity.ys.copy_(torch.tensor(new_ys))
            #print("NEW", saliency_map_processing.nonlinearity.ys)

        # set center bias
        if num_centerbias > 0:
            #print("OLD CB", saliency_map_processing.centerbias.nonlinearity.ys)
            new_centerbias = np.linspace(0, -2, num_centerbias)
            saliency_map_processing.centerbias.nonlinearity.ys.copy_(torch.tensor(new_centerbias))
            #print("NEW CB", saliency_map_processing.centerbias.nonlinearity.ys)

        if has_alpha:
            saliency_map_processing.centerbias.alpha.copy_(torch.tensor(0.83))

        if is_blurring:
            saliency_map_processing.blur.sigma.copy_(torch.tensor(4.0))

    return SaliencyMapProcessingModel(
        saliency_map_model=saliency_model,
        saliency_map_processing=saliency_map_processing,
        saliency_min=0,
        saliency_max=1,
    )


@pytest.fixture
def fixations(stimuli, probabilistic_model):
    return probabilistic_model.sample(stimuli, 1000, rst=np.random.RandomState(seed=42))


def test_optimize_for_information_gain(stimuli, fixations, saliency_model, probabilistic_model, is_blurring, num_nonlinearity, has_alpha, num_centerbias):

    if num_centerbias == 0 and has_alpha:
        pytest.skip("parameter combination doesn't make sense")

    expected_information_gain = probabilistic_model.information_gain(stimuli, fixations, average='image')

    optimize = []
    if num_nonlinearity > 0:
        optimize.append('nonlinearity')

    if num_centerbias > 0:
        optimize.append('centerbias')

    if has_alpha:
        optimize.append('alpha')

    if is_blurring:
        blur_radius = 1.0
        optimize.append('blur_radius')
    else:
        blur_radius = 0.0

    if not optimize:
        return

    model1, ret1 = optimize_for_information_gain(
        saliency_model,
        stimuli,
        fixations,
        average='fixations',
        saliency_min=0,
        saliency_max=1,
        verbose=2,
        batch_size=10,
        minimize_options={'verbose': 10},
        maxiter=500,
        num_nonlinearity=num_nonlinearity,
        num_centerbias=num_centerbias,
        blur_radius=blur_radius,
        optimize=optimize,
        return_optimization_result=True,
        framework='torch',
    )

    assert ret1.status in [
        0,  # success
        9,  # max iter reached
    ]

    reached_information_gain = model1.information_gain(stimuli, fixations, average='image')

    #print(expected_information_gain, reached_information_gain)
    assert reached_information_gain >= expected_information_gain - 0.001
    assert reached_information_gain <= expected_information_gain + 0.001