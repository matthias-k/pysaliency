from hypothesis import given, strategies as st
import numpy as np

from pysaliency.numba_utils import auc_for_one_positive
from pysaliency.roc import general_roc


def test_auc_for_one_positive():
    assert auc_for_one_positive(1, [0, 2]) == 0.5
    assert auc_for_one_positive(1, [1]) == 0.5
    assert auc_for_one_positive(3, [0]) == 1.0
    assert auc_for_one_positive(0, [3]) == 0.0


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1), st.floats(allow_nan=False, allow_infinity=False))
def test_simple_auc_hypothesis(negatives, positive):
    old_auc, _, _ = general_roc(np.array([positive]), np.array(negatives))
    new_auc = auc_for_one_positive(positive, np.array(negatives))
    np.testing.assert_allclose(old_auc, new_auc)
