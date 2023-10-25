#%%cython
# Circumvents a bug(?) in cython:
# http://stackoverflow.com/a/13976504
STUFF = "Hi"


import numpy as np
cimport numpy as np
cimport cython


#Do not check for index errors
@cython.boundscheck(False)
#Do not enable negativ indices
@cython.wraparound(False)
#Use native c division
@cython.cdivision(True)
def real_ROC(image, fixation_data, int judd=0):
    fixations_orig = np.zeros_like(image)
    fixations_orig[fixation_data] = 1.0
    image_1d = image.flatten()
    cdef np.ndarray[double, ndim=1] fixations = fixations_orig.flatten()
    inds = image_1d.argsort()
    #image_1d = image_1d[inds]
    fixations = fixations[inds]
    cdef np.ndarray[double, ndim=1] image_sorted = image_1d[inds]
    cdef np.ndarray[double, ndim=1] fixation_values_sorted = image[fixation_data]
    fixation_values_sorted.sort()
    cdef int i
    cdef int N = image_1d.shape[0]
    cdef int fix_count = fixations.sum()
    cdef int false_count = N-fix_count
    cdef int correct_count = 0
    cdef int false_positive_count = 0
    cdef int length
    if judd:
        length = fix_count+2
        assert len(fixation_values_sorted) == fix_count
    else:
        length = N+1
    cdef np.ndarray[double, ndim=1] precs = np.zeros(length)
    cdef np.ndarray[double, ndim=1] false_positives = np.zeros(length)
    for i in range(N):
        #print fixations[N-i-1],
        #print image_1d[N-i-1]
        # Every pixel is a nonfixation
        false_positive_count += 1
        if fixations[N-i-1]:
            correct_count += 1
            if judd:
                if i == N - 1 or fixation_values_sorted[N-i-1] != fixation_values_sorted[N-i-2]:
                    precs[correct_count] = float(correct_count)/fix_count
                    false_positives[correct_count] = float(false_positive_count)/false_count
                else:
                    precs[correct_count] = precs[correct_count - 1]
                    false_positives[correct_count] = false_positives[correct_count - 1]
        if not judd:
            if i == N-1 or image_sorted[N-i-1] != image_sorted[N-i-2]:
                precs[i+1] = float(correct_count)/fix_count
                false_positives[i+1] = float(false_positive_count)/false_count
            else:
                precs[i+1] = precs[i]
                false_positives[i+1] = false_positives[i]
        #print false_positives[i+1]
    precs[length-1] = 1.0
    false_positives[length-1] = 1.0
    aoc = np.trapz(precs, false_positives)
    return aoc, precs, false_positives


#Do not check for index errors
@cython.boundscheck(False)
#Do not enable negativ indices
@cython.wraparound(False)
#Use native c division
@cython.cdivision(True)
def general_roc(np.ndarray[double, ndim=1] positives, np.ndarray[double, ndim=1] negatives, int judd=0):
    """calculate ROC score for given values of positive and negative
    distribution"""
    cdef np.ndarray[double, ndim=1] sorted_positives = np.sort(positives)[::-1]
    cdef np.ndarray[double, ndim=1] sorted_negatives = np.sort(negatives)[::-1]
    cdef np.ndarray[double, ndim=1] all_values
    if judd == 0:
        all_values = np.hstack([positives, negatives])
        all_values = np.sort(all_values)[::-1]
    else:
        min_val = np.min([sorted_positives[len(positives)-1], sorted_negatives[len(negatives)-1]])
        max_val = np.max([sorted_positives[0], sorted_negatives[0]])+1
        all_values = np.hstack((max_val, positives, min_val))
        all_values = np.sort(all_values)[::-1]
    cdef np.ndarray[double, ndim=1] false_positive_rates = np.zeros(len(all_values)+1)
    cdef np.ndarray[double, ndim=1] hit_rates = np.zeros(len(all_values)+1)
    cdef int true_positive_count = 0
    cdef int false_positive_count = 0
    cdef int positive_count = len(positives)
    cdef int negative_count = len(negatives)
    cdef int i
    cdef double theta
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


#Do not check for index errors
@cython.boundscheck(False)
#Do not enable negativ indices
@cython.wraparound(False)
#Use native c division
@cython.cdivision(True)
def general_rocs_per_positive(np.ndarray[double, ndim=1] positives, np.ndarray[double, ndim=1] negatives):
    """calculate ROC scores for each positive against a list of negatives
    distribution. The mean over the result will equal the return value of `general_roc`."""
    cdef np.ndarray[double, ndim=1] sorted_positives = np.sort(positives)
    cdef np.ndarray[double, ndim=1] sorted_negatives = np.sort(negatives)
    cdef np.ndarray[long, ndim=1] sorted_inds = np.argsort(positives)

    cdef np.ndarray[double, ndim=1] results = np.empty(len(positives))
    cdef int true_positive_count = 0
    cdef int false_positive_count = 0
    cdef int true_negative_count = 0
    cdef int positive_count = len(positives)
    cdef int negative_count = len(negatives)
    cdef int i
    cdef double last_theta = -np.inf
    cdef double theta

    cdef int true_negatives_count = 0
    cdef int equal_count = 0
    for i in range(len(sorted_positives)):
        theta = sorted_positives[i]
        #print('theta', theta)
        if theta == last_theta:
            #print('same')
            results[sorted_inds[i]] = (1.0*true_negatives_count + 0.5*equal_count) / negative_count
            continue

        true_negatives_count = true_negatives_count + equal_count

        while true_negatives_count < negative_count and sorted_negatives[true_negatives_count] < theta:
            true_negatives_count += 1
            #print('.')
        equal_count = 0
        while true_negatives_count + equal_count < negative_count and sorted_negatives[true_negatives_count+equal_count] <= theta:
            equal_count += 1
            #print('=')
        results[sorted_inds[i]] = (1.0*true_negatives_count + 0.5*equal_count) / negative_count

        last_theta = theta
    return results
