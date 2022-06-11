from __future__ import print_function, unicode_literals, division, absolute_import

import numba
import numpy as np


def fill_fixation_map(fixation_map, fixations, check_bounds=True):
    if check_bounds:
        if np.any(fixations < 0):
            raise ValueError("Negative fixation positions!")
        if np.any(fixations[:, 0] >= fixation_map.shape[0]):
            raise ValueError("Fixations y positions out of bound!")
        if np.any(fixations[:, 1] >= fixation_map.shape[1]):
            raise ValueError("Fixations x positions out of bound!")
    return _fill_fixation_map(fixation_map, fixations)


@numba.jit(nopython=True)
def _fill_fixation_map(fixation_map, fixations):
    """fixationmap: 2d array. fixations: Nx2 array of y, x positions"""
    for i in range(len(fixations)):
        fixation_y, fixation_x = fixations[i]
        fixation_map[int(fixation_y), int(fixation_x)] += 1


def auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.

    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    return _auc_for_one_positive(positive, np.asarray(negatives))


@numba.jit(nopython=True)
def _auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.

    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    count = 0
    for negative in negatives:
        if negative < positive:
            count += 1
        elif negative == positive:
            count += 0.5

    return count / len(negatives)
