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


def general_roc_numba(positives, negatives, judd=0):
    sorted_positives = np.sort(positives)[::-1]
    sorted_negatives = np.sort(negatives)[::-1]

    if judd == 0:
        all_values = np.hstack([positives, negatives])
        all_values = np.sort(all_values)[::-1]
    else:
        min_val = min(sorted_positives[len(positives)-1], sorted_negatives[len(negatives)-1])
        max_val = max(sorted_positives[0], sorted_negatives[0]) + 1
        all_values = np.hstack((max_val, positives, min_val))
        all_values = np.sort(all_values)[::-1]

    false_positive_rates = np.zeros(len(all_values) + 1)
    hit_rates = np.zeros(len(all_values) + 1)
    positive_count = len(positives)
    negative_count = len(negatives)
    return _general_roc_numba(all_values, sorted_positives, sorted_negatives, positive_count, negative_count, false_positive_rates, hit_rates)


@numba.jit(nopython=True)
def _general_roc_numba(all_values, sorted_positives, sorted_negatives, positive_count, negative_count, false_positive_rates, hit_rates):
    """calculate ROC score for given values of positive and negative
    distribution"""

    true_positive_count = 0
    false_positive_count = 0
    for i in range(len(all_values)):
        theta = all_values[i]
        while true_positive_count < positive_count and sorted_positives[true_positive_count] >= theta:
            true_positive_count += 1
        while false_positive_count < negative_count and sorted_negatives[false_positive_count] >= theta:
            false_positive_count += 1
        false_positive_rates[i+1] = float(false_positive_count) / negative_count
        hit_rates[i+1] = float(true_positive_count) / positive_count

    auc = np.trapz(hit_rates, false_positive_rates)
    return auc, hit_rates, false_positive_rates
