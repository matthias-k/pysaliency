import numpy as np
from packaging import version
from scipy.ndimage import gaussian_filter as scipy_filter
import torch

import hypothesis
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hypothesis_np
import pytest

from pysaliency.torch_utils import gaussian_filter, gaussian_filter_1d_new_torch, gaussian_filter_1d_old_torch


@pytest.fixture(params=[20.0])
def sigma(request):
    return request.param


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request):
    return request.param


def test_gaussian_filter(sigma, dtype):
    #window_radius = int(sigma*4)
    test_data = 10*np.ones((4, 1, 100, 100))
    test_data += np.random.randn(4, 1, 100, 100)

    test_tensor = torch.tensor(test_data, dtype=dtype)
    
    output = gaussian_filter(
        tensor=test_tensor,
        sigma=torch.tensor(sigma),
        truncate=4,
        dim=[2, 3],
    ).detach().cpu().numpy()[0, 0, :, :]
    
    scipy_out = scipy_filter(test_data[0, 0], sigma, mode='nearest')

    if dtype == torch.float32:
        rtol = 5e-6
    else:
        rtol = 1e-7
    np.testing.assert_allclose(output, scipy_out, rtol=rtol)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('1.7')  # new code doesn't work because no `torch.movedim`
    or version.parse(torch.__version__) >= version.parse('1.11'),  # old code doesn't work because torch's conv1d got stricter about input shape
    reason="torch either too new for old implementation or too old for new implementation"
)
@given(hypothesis_np.arrays(
    dtype=hypothesis_np.floating_dtypes(sizes=(32, 64), endianness='='),
    shape=st.tuples(
        st.integers(min_value=1, max_value=100),
        st.just(1),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100)
    )),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=50),
    st.integers(min_value=2, max_value=3),
)
#@hypothesis.settings(verbosity=hypothesis.Verbosity.verbose)
@hypothesis.settings(deadline=5000)
def test_compare_gaussian_1d_implementations(data, sigma, dim):
    data_tensor = torch.tensor(data)
    old_data = gaussian_filter_1d_old_torch(data_tensor, sigma=sigma, dim=dim).detach().cpu().numpy()
    new_data = gaussian_filter_1d_new_torch(data_tensor, sigma=sigma, dim=dim).detach().cpu().numpy()

    np.testing.assert_allclose(old_data, new_data)