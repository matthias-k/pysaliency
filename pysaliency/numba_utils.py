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
    hit_rates, false_positive_rates = _general_roc_numba(all_values, sorted_positives, sorted_negatives, false_positive_rates, hit_rates)
    auc = np.trapz(hit_rates, false_positive_rates)

    return auc, hit_rates, false_positive_rates


@numba.jit(nopython=True)
def _general_roc_numba(all_values, sorted_positives, sorted_negatives, false_positive_rates, hit_rates):
    """calculate ROC score for given values of positive and negative
    distribution"""

    positive_count = len(sorted_positives)
    negative_count = len(sorted_negatives)
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

    return hit_rates, false_positive_rates


def general_rocs_per_positive_numba(positives, negatives):
    sorted_positives = np.sort(positives)
    sorted_negatives = np.sort(negatives)
    sorted_inds = np.argsort(positives)

    results = np.empty(len(positives))
    results = _general_rocs_per_positive_numba(sorted_positives, sorted_negatives, sorted_inds, results)

    return results


@numba.jit(nopython=True)
def _general_rocs_per_positive_numba(sorted_positives, sorted_negatives, sorted_inds, results):
    """calculate ROC scores for each positive against a list of negatives
    distribution. The mean over the result will equal the return value of `general_roc`."""

    true_negatives_count = 0
    equal_count = 0
    last_theta = -np.inf
    negative_count = len(sorted_negatives)

    for i, theta in enumerate(sorted_positives):

        if theta == last_theta:
            results[sorted_inds[i]] = (1.0 * true_negatives_count + 0.5 * equal_count) / negative_count
            continue

        true_negatives_count = true_negatives_count + equal_count

        while true_negatives_count < negative_count and sorted_negatives[true_negatives_count] < theta:
            true_negatives_count += 1

        equal_count = 0
        while true_negatives_count + equal_count < negative_count and sorted_negatives[true_negatives_count + equal_count] <= theta:
            equal_count += 1

        results[sorted_inds[i]] = (1.0 * true_negatives_count + 0.5 * equal_count) / negative_count

        last_theta = theta

    return results