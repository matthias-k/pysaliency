from __future__ import division, print_function

import numpy as np


#def create_segments(folds):
#    folds = sorted(folds)
#    segments = []
#    for f in folds:
#        if not segments or segments[-1][-1] < f:
#            segments.append([f, f+1])
#        else:
#            segments[-1][-1] = f + 1
#    return segments
#
#
#def simplify_segment(segment):
#    if not isinstance(segment, (list, tuple)):
#        return segment
#    if segment[-1] == segment[0]+1:
#        return segment[0]
#    else:
#        return segment
#
#
#def format_segment(segment):
#    segment = simplify_segment(segment)
#    if isinstance(segment, list):
#        return "{}:{}".format(*segment)
#    else:
#        return "{}".format(segment)
#
#
#def format_segments(segments):
#    parts = [format_segment(s) for s in segments]
#    return ",".join(parts)
#
#
#def format_fold(fold):
#    return format_segments(create_segments(fold))
#
#
#def create_train_folds(crossval_folds, val_folds, test_folds):
#    all_folds = list(range(crossval_folds))
#    if isinstance(val_folds, int):
#        val_folds = [val_folds]
#    if isinstance(test_folds, int):
#        test_folds = [test_folds]
#
#    train_folds = [f for f in all_folds if not (f in val_folds or f in test_folds)]
#
#    #train_segments, val_segments, test_segments = create_segments(train_folds), create_segments(val_folds), create_segments(test_folds)
#    #return format_segments(train_segments), format_segments(val_segments), format_segments(test_segments)
#    return train_folds, val_folds, test_folds
#
#
#def get_crossval_folds(crossval_folds, crossval_no, test_folds=1, val_folds=1):
#    assert test_folds <= 1
#    if test_folds:
#        _test_folds = [crossval_no]
#        _val_folds = [(crossval_no - i - 1) % crossval_folds for i in range(val_folds)]
#
#    else:
#        assert val_folds == 1
#
#        _test_folds = [crossval_no]
#        _val_folds = [crossval_no]
#
#    _train_folds, _val_folds, _test_folds = create_train_folds(crossval_folds, _val_folds, _test_folds)
#
#    return _train_folds, _val_folds, _test_folds
#
#
#def get_crossval_postfixes(crossval_folds, crossval_no, test_folds=1, val_folds=1, split_random=True):
#    """ Create filter postfixes for crossvalidation
#
#        if test_folds == 0, validation will be used as test
#    """
#
#    _train_folds, _val_folds, _test_folds = get_crossval_folds(crossval_folds, crossval_no, test_folds=test_folds, val_folds=val_folds)
#
#    if split_random:
#        split_type = 'splitstimulirandom'
#    else:
#        split_type = 'splitstimuli'
#
#    train_filter = "{}_{}_{}".format(split_type, crossval_folds, format_fold(_train_folds))
#    val_filter = "{}_{}_{}".format(split_type, crossval_folds, format_fold(_val_folds))
#    test_filter = "{}_{}_{}".format(split_type, crossval_folds, format_fold(_test_folds))
#
#    return train_filter, val_filter, test_filter


def parse_list_of_intervals(description):
    """parses a string as "1.0:3.0,5.0:5.6,7" into [(1.0, 3.0), (5.0, 5.6), (7,)]
    """
    intervals = description.split(',')
    return [parse_interval(interval) for interval in intervals]


def parse_interval(interval):
    parts = interval.split(':')
    if len(parts) not in [1, 2]:
        raise ValueError("Invalid interval", interval)
    return tuple([float(part.strip()) for part in parts])


def filter_fixations_by_number(fixations, intervals):
    intervals = _check_intervals(intervals, type=int)
    inds = np.zeros_like(fixations.x, dtype=bool)

    for n1, n2 in intervals:
        _inds = np.logical_and(fixations.lengths >= n1, fixations.lengths < n2)
        inds = np.logical_or(inds, _inds)

    return fixations[inds]


def _check_intervals(intervals, type=float):
    if isinstance(intervals, (float, int)):
        intervals = [intervals]

    new_intervals = []
    for interval in intervals:
        new_intervals.append(_check_interval(interval, type=type))
    return new_intervals


def _check_interval(interval, type=float):
    if isinstance(interval, (float, int)):
        interval = [interval]

    if len(interval) == 1:
        if type != int:
            raise ValueError("single-value intervals only allowed for integer data!")
        interval = [interval[0], interval[0] + 1]

    if len(interval) != 2:
        raise ValueError("Intervals need two values", interval)
    new_interval = []
    for value in interval:
        if type(value) != value:
            raise ValueError("Invalid value for this type", value, type)
        new_interval.append(type(value))

    return tuple(new_interval)
