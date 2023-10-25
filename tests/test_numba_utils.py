from hypothesis import given, strategies as st, assume, settings
import numpy as np

from pysaliency.numba_utils import auc_for_one_positive, general_roc_numba, general_rocs_per_positive_numba
from pysaliency.roc_cython import general_roc, general_rocs_per_positive


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


@settings(deadline=None)        #to remove time limit from a test
@given(st.lists(st.floats(allow_infinity=False,allow_nan=False),min_size=1), st.lists(st.floats(allow_infinity=False,allow_nan=False),min_size=1))
def test_numba_auc_test1(positives,negatives):
    positives = np.array(positives)
    negatives = np.array(negatives)
    numba_output = general_roc_numba(positives,negatives)
    cython_output = general_roc(positives,negatives)
    assert np.isclose(numba_output[0],cython_output[0])
    assert (numba_output[1] == cython_output[1]).all()
    assert (numba_output[2] == cython_output[2]).all()


@settings(deadline=None)
@given(st.lists(st.floats(allow_infinity=False,allow_nan=False),min_size=1), st.floats(allow_infinity=False,allow_nan=False))
def test_numba_auc_test2(positives,temp_variable):
    positives = np.array(positives)
    negatives = positives+temp_variable
    numba_output = general_roc_numba(positives,negatives)
    cython_output = general_roc(positives,negatives)
    assert np.isclose(numba_output[0],cython_output[0])
    assert (numba_output[1] == cython_output[1]).all()
    assert (numba_output[2] == cython_output[2]).all()


@settings(deadline=None)
@given(st.lists(st.floats(allow_infinity=False,allow_nan=False),min_size=1), st.lists(st.floats(allow_infinity=False,allow_nan=False),min_size=1))
def test_numba_rocs_per_positive(positives,negatives):
    positives = np.array(positives)
    negatives = np.array(negatives)
    numba_output = general_rocs_per_positive_numba(positives,negatives)
    cython_output = general_rocs_per_positive(positives,negatives)
    assert (numba_output == cython_output).all()