import numpy as np
import pytest

from pysaliency.saliency_map_conversion import optimize_for_information_gain
from pysaliency import Stimuli, Fixations, GaussianSaliencyMapModel


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

    assert res.status in [
        0,  # success
        9,  # max iter reached
    ]

    assert smc
