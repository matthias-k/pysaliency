import numpy as np
from scipy.ndimage import gaussian_filter as scipy_filter
import torch

import pytest

from pysaliency.torch_utils import gaussian_filter


@pytest.fixture(params=[20.0])
def sigma(request):
    return request.param


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request):
    return request.param


def test_gaussian_filter(sigma, dtype):
    #window_radius = int(sigma*4)
    test_data = 10*np.ones((100, 100))
    test_data += np.random.randn(100, 100)

    test_tensor = torch.tensor([test_data], dtype=dtype)
    
    output = gaussian_filter(
        tensor=test_tensor,
        sigma=torch.tensor(sigma),
        truncate=4,
        dim=[1, 2],
    ).detach().cpu().numpy()
    
    scipy_out = scipy_filter(test_data, sigma, mode='nearest')

    if dtype == torch.float32:
        rtol = 5e-6
    else:
        rtol = 1e-7
    np.testing.assert_allclose(output, scipy_out, rtol=rtol)