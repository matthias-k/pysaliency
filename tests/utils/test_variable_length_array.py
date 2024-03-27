import numpy as np
import pytest

from pysaliency.utils import build_padded_2d_array
from pysaliency.utils.variable_length_array import VariableLengthArray


def test_variable_length_array_from_padded_array_basics():
    # Test case 1
    data = build_padded_2d_array([[1.0, 2, 3], [4, 5]])
    lengths = np.array([3, 2])
    array = VariableLengthArray(data, lengths)

    assert len(array) == 2

    rows = list(array)
    assert np.array_equal(rows[0], np.array([1, 2, 3]))
    assert np.array_equal(rows[1], np.array([4, 5]))

def test_variable_length_array_from_padded_array():
    # Test case 1
    data = build_padded_2d_array([[1.0, 2, 3], [4, 5]])
    lengths = np.array([3, 2])
    array = VariableLengthArray(data, lengths)

    # test accessing rows
    assert np.array_equal(array[0], np.array([1, 2, 3]))
    assert np.array_equal(array[1], np.array([4, 5]))

    # test accessing elements
    assert np.array_equal(array[0, 1], 2)

    # acessing elements outside the length of the row should raise an IndexError
    with pytest.raises(IndexError):
        array[1, 2]

    # test slicing
    assert np.array_equal(array[:, 0], [1, 4])

    # test slicing with negative indices
    assert np.array_equal(array[:, -1], [3, 5])



    # Test case 2
    data = build_padded_2d_array([[1.0, 2], [3, 4, 5]])
    lengths = np.array([2, 3])
    array = VariableLengthArray(data, lengths)

    # test accessing rows
    assert np.array_equal(array[0], np.array([1, 2]))
    assert np.array_equal(array[1], np.array([3, 4, 5]))

    # test accessing elements
    assert np.array_equal(array[0, 1], 2)
    assert np.array_equal(array[1, 2], 5)

    # acessing elements outside the length of the row should raise an IndexError
    with pytest.raises(IndexError):
        array[1, 3]

    # test slicing
    assert np.array_equal(array[:, 0], [1, 3])

    # test slicing with negative indices
    assert np.array_equal(array[:, -1], [2, 5])


def test_variable_length_array_slicing_with_slices():
    data = build_padded_2d_array([[1.0, 2, 3], [4, 5], [6, 7, 8, 9]])
    lengths = np.array([3, 2, 4])
    array = VariableLengthArray(data, lengths)

    sub_array = array[1:]
    assert isinstance(sub_array, VariableLengthArray)
    assert len(sub_array) == 2
    np.testing.assert_array_equal(sub_array._data, data[1:])
    np.testing.assert_array_equal(sub_array[0], np.array([4, 5]))
    np.testing.assert_array_equal(sub_array[1], np.array([6, 7, 8, 9]))

    sub_array = array[:2]
    assert isinstance(sub_array, VariableLengthArray)
    assert len(sub_array) == 2
    np.testing.assert_array_equal(sub_array._data, data[:2])  # one length item is cut off
    np.testing.assert_array_equal(sub_array[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(sub_array[1], np.array([4, 5]))


def test_variable_length_array_slicing_with_indices():
    data = build_padded_2d_array([[1.0, 2, 3], [4, 5], [6, 7, 8, 9]])
    lengths = np.array([3, 2, 4])
    array = VariableLengthArray(data, lengths)

    sub_array = array[[0, 2]]
    assert isinstance(sub_array, VariableLengthArray)
    assert len(sub_array) == 2
    np.testing.assert_array_equal(sub_array._data, data[[0, 2]])
    np.testing.assert_array_equal(sub_array[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(sub_array[1], np.array([6, 7, 8, 9]))


def test_variable_length_array_slicing_with_mask():
    data = build_padded_2d_array([[1.0, 2, 3], [4, 5], [6, 7, 8, 9]])
    lengths = np.array([3, 2, 4])
    array = VariableLengthArray(data, lengths)

    sub_array = array[[True, False, True]]
    assert isinstance(sub_array, VariableLengthArray)
    assert len(sub_array) == 2
    np.testing.assert_array_equal(sub_array._data, data[[0, 2]])
    np.testing.assert_array_equal(sub_array[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(sub_array[1], np.array([6, 7, 8, 9]))


def test_variable_length_array_from_list_of_arrays():
    # Test case 1
    data = [[1, 2, 3], [4, 5]]
    lengths = np.array([3, 2])
    array = VariableLengthArray(data, lengths)

    np.testing.assert_array_equal(array._data, np.array([[1, 2, 3], [4, 5, np.nan]]))


def test_variable_length_array_from_list_of_arrays_without_specified_lengths():
    data = [[1, 2, 3], [4, 5]]
    lengths = np.array([3, 2])
    array = VariableLengthArray(data)

    np.testing.assert_array_equal(array._data, np.array([[1, 2, 3], [4, 5, np.nan]]))
    np.testing.assert_array_equal(array.lengths, lengths)


def test_variable_length_array_inconsistent_lengths():
    # consistent case
    data = [[1, 2, 3], [4]]
    lengths = np.array([3, 1])

    VariableLengthArray(data, lengths)

    # inconsistent case
    data = [[1, 2, 3], [4]]
    lengths = np.array([3, 2])

    with pytest.raises(ValueError):
        VariableLengthArray(data, lengths)