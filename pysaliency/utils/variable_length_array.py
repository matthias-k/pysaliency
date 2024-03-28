from typing import Optional, Union, List

import numpy as np

from . import build_padded_2d_array


class VariableLengthArray:
    """
    Represents a variable length array.

    The following indexing operations are supported:
    - Accessing rows: array[i]
    - Accessing elements: array[i, j] where j can also be negative to get elements from the end of the row
    - Slicing: array[i:j, k] where k can also be negative to get elements from the end of each row


    Args:
        data (Union[np.ndarray, list[list]]): The data for the array. Can be either a numpy array or a list of lists.
        lengths (np.ndarray): The lengths of each row in the data array.

    Attributes:
        _data (np.ndarray): The internal data array with padded rows.
        lengths (np.ndarray): The lengths of each row in the data array.

    Methods:
        __len__(): Returns the number of rows in the array.
        __getitem__(index): Returns the value(s) at the specified index(es) in the array.
    """

    def __init__(self, data: Union[np.ndarray, List[list]], lengths: Optional[np.ndarray] = None):
        """List
        Initialize the VariableLengthArray object.

        Args:
            data (Union[np.ndarray, list[list]]): The input data, which can be either a numpy array or a list of lists.
            lengths (np.ndarray): An array containing the lengths of each row in the data.

        Raises:
            ValueError: If the input data shape doesn't match the provided lengths.

        """

        if lengths is not None:
            if len(data) != len(lengths):
                raise ValueError(f"The number of rows in the data array has to match the number of elements in lengths ({len(data)} != {len(lengths)})")

            if not isinstance(data, np.ndarray):
                for row, length in zip(data, lengths):
                    if len(row) != length:
                        raise ValueError(f"The length of row {row} does not match the specified length {length}")
            else:
                if not data.ndim >= 2:
                    raise ValueError("If data is a numpy array, it has to be at least 2-dimensional")
                if np.max(lengths) > data.shape[1]:
                    raise ValueError("The specified lengths are larger than the number of columns in the data array")

        else:
            if isinstance(data, np.ndarray):
                raise ValueError("If data is a numpy array, lengths must be provided")
            lengths = np.array([len(row) for row in data])

        if isinstance(data, np.ndarray):
            self._data = data
        else:
            self._data = build_padded_2d_array(data, max_length=np.max(lengths))

        # max_len = np.max(lengths)
        # self._data = np.full((len(data), max_len), np.nan)
        # for i, row in enumerate(data):
        #     if len(row) < lengths[i]:
        #         raise ValueError(f"Row {i} has fewer elements than specified in lengths ({len(row)} < {lengths[i]}")
        #     self._data[i, :lengths[i]] = row[:lengths[i]]
        self.lengths = lengths

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row_idx, col_idx = index
            if isinstance(row_idx, slice):
                if isinstance(col_idx, int):
                    return np.array([self._data[i, :self.lengths[i]][col_idx] for i in range(*row_idx.indices(len(self._data)))])
                elif isinstance(col_idx, slice):
                    # does this work?
                    return self._data[row_idx, :self.lengths[row_idx]][col_idx]
            else:
                return self._data[row_idx, :self.lengths[row_idx]][col_idx]
        elif isinstance(index, int):
            return self._data[index, :self.lengths[index]]
        else:
            return VariableLengthArray(self._data[index], self.lengths[index])
            # new_lengths = self.lengths[index]
            # max_length = np.max(new_lengths)
            # new_data = self._data[index, :max_length]
            # return VariableLengthArray(new_data, new_lengths)

    def copy(self):
        return VariableLengthArray(self._data.copy(), self.lengths.copy())


def concatenate_variable_length_arrays(arrays: List[VariableLengthArray]):
    """
    Concatenate a list of VariableLengthArray objects along the first axis.

    Args:
        arrays (List[VariableLengthArray]): List of VariableLengthArray objects to concatenate.

    Returns:
        VariableLengthArray: The concatenated VariableLengthArray object.
    """
    lengths = np.concatenate([array.lengths for array in arrays])

    datas = [array._data for array in arrays]
    max_cols = max(a.shape[1] for a in datas)
    padded_datas = []
    for a in datas:
        if a.shape[1] < max_cols:
            padding = np.empty((a.shape[0], max_cols-a.shape[1]), dtype=a.dtype)
            padding[:] = np.nan
            padded_datas.append(np.hstack((a, padding)))
        else:
            padded_datas.append(a)
    data = np.vstack(padded_datas)

    return VariableLengthArray(data, lengths)
