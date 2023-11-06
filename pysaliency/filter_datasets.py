from __future__ import division, print_function

import numpy as np

from boltons.iterutils import chunked

from .datasets import create_subset, FixationTrains, Fixations, Stimuli


def train_split(stimuli, fixations, crossval_folds, fold_no, val_folds=1, test_folds=1, random=True, stratified_attributes=None):
    return crossval_split(stimuli, fixations, crossval_folds, fold_no, val_folds=val_folds, test_folds=test_folds, random=random, split='train', stratified_attributes=stratified_attributes)


def validation_split(stimuli, fixations, crossval_folds, fold_no, val_folds=1, test_folds=1, random=True, stratified_attributes=None):
    return crossval_split(stimuli, fixations, crossval_folds, fold_no, val_folds=val_folds, test_folds=test_folds, random=random, split='val', stratified_attributes=stratified_attributes)


def test_split(stimuli, fixations, crossval_folds, fold_no, val_folds=1, test_folds=1, random=True, stratified_attributes=None):
    return crossval_split(stimuli, fixations, crossval_folds, fold_no, val_folds=val_folds, test_folds=test_folds, random=random, split='test', stratified_attributes=stratified_attributes)


def crossval_splits(stimuli, fixations, crossval_folds, fold_no, val_folds=1, test_folds=1, random=True, stratified_attributes=None):
    return (
        crossval_split(stimuli, fixations, crossval_folds, fold_no, val_folds=val_folds, test_folds=test_folds, random=random, split='train', stratified_attributes=stratified_attributes),
        crossval_split(stimuli, fixations, crossval_folds, fold_no, val_folds=val_folds, test_folds=test_folds, random=random, split='val', stratified_attributes=stratified_attributes),
        crossval_split(stimuli, fixations, crossval_folds, fold_no, val_folds=val_folds, test_folds=test_folds, random=random, split='test', stratified_attributes=stratified_attributes),
    )


def crossval_split(stimuli, fixations, crossval_folds, fold_no, val_folds=1, test_folds=1, random=True, split='train', stratified_attributes=None):
    train_folds, val_folds, test_folds = get_crossval_folds(crossval_folds, fold_no, test_folds=test_folds, val_folds=val_folds)

    if split == 'train':
        folds = train_folds
    elif split == 'val':
        folds = val_folds
    elif split == 'test':
        folds = test_folds
    else:
        raise ValueError(split)

    return _get_crossval_split(stimuli, fixations, crossval_folds, included_splits=folds, random=random, stratified_attributes=stratified_attributes)


def _get_crossval_split(stimuli, fixations, split_count, included_splits, random=True, stratified_attributes=None):
    if stratified_attributes is not None:
        return _get_stratified_crossval_split(stimuli, fixations, split_count, included_splits, random=random, stratified_attributes=stratified_attributes)

    inds = list(range(len(stimuli)))
    if random:
        print("Using random shuffles for crossvalidation")
        rst = np.random.RandomState(seed=42)
        rst.shuffle(inds)
        inds = list(inds)
    size = int(np.ceil(len(inds) / split_count))
    chunks = chunked(inds, size=size)

    inds = []
    for split_nr in included_splits:
        inds.extend(chunks[split_nr])

    stimuli, fixations = create_subset(stimuli, fixations, inds)
    return stimuli, fixations


def _get_stratified_crossval_split(stimuli, fixations, split_count, included_splits, random=True, stratified_attributes=None):
    from sklearn.model_selection import StratifiedKFold
    labels = []
    for attribute_name in stratified_attributes:
        attribute_data = np.array(stimuli.attributes[attribute_name])
        if attribute_data.ndim == 1:
            attribute_data = attribute_data[:, np.newaxis]
        labels.append(attribute_data)
    labels = np.hstack(labels)

    if labels.shape[1] > 1:
        # StratifiedKFold doesn't support multiple labels
        final_label_dict = {}
        final_labels = []
        for label in labels:
            label = tuple(label)
            if label not in final_label_dict:
                final_label_dict[label] = len(final_label_dict)
            final_labels.append(final_label_dict[label])

        labels = np.array(final_labels)

    X = np.ones((len(stimuli), 1))

    rst = np.random.RandomState(42)

    inds = []
    k_fold = StratifiedKFold(n_splits=split_count, shuffle=random, random_state=rst)
    for i, (train_index, test_index) in enumerate(k_fold.split(X, labels)):
        if i in included_splits:
            inds.extend(test_index)

    stimuli, fixations = create_subset(stimuli, fixations, inds)
    return stimuli, fixations


def create_train_folds(crossval_folds, val_folds, test_folds):
    all_folds = list(range(crossval_folds))
    if isinstance(val_folds, int):
        val_folds = [val_folds]
    if isinstance(test_folds, int):
        test_folds = [test_folds]

    train_folds = [f for f in all_folds if not (f in val_folds or f in test_folds)]

    return train_folds, val_folds, test_folds


def get_crossval_folds(crossval_folds, crossval_no, test_folds=1, val_folds=1):
    assert test_folds <= 1
    if test_folds:
        _test_folds = [crossval_no]
        _val_folds = [(crossval_no - i - 1) % crossval_folds for i in range(val_folds)]

    else:
        assert val_folds == 1

        _test_folds = [crossval_no]
        _val_folds = [crossval_no]

    _train_folds, _val_folds, _test_folds = create_train_folds(crossval_folds, _val_folds, _test_folds)

    return _train_folds, _val_folds, _test_folds


def iterate_crossvalidation(stimuli, fixations, crossval_folds, val_folds=1, test_folds=1, random=True, stratified_attributes=None):
    """iterate over crossvalidation folds. Each fold will yield
          train_stimuli, train_fixations, val_, test_stimuli, test_fixations
    """
    kwargs = {
        'crossval_folds': crossval_folds,
        'val_folds': val_folds,
        'test_folds': test_folds,
        'random': random,
        'stratified_attributes': stratified_attributes,
    }

    for fold_no in range(crossval_folds):
        train_stimuli, train_fixations = train_split(
            stimuli, fixations,
            fold_no=fold_no,
            **kwargs)
        val_stimuli, val_fixations = validation_split(
            stimuli, fixations,
            fold_no=fold_no,
            **kwargs)
        test_stimuli, test_fixations = test_split(
            stimuli, fixations,
            fold_no=fold_no,
            **kwargs)

        yield train_stimuli, train_fixations, val_stimuli, val_fixations, test_stimuli, test_fixations


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


def filter_stimuli_by_number(stimuli, fixations, intervals):
    intervals = _check_intervals(intervals, type=int)
    mask = np.zeros(len(stimuli), dtype=bool)

    for n1, n2 in intervals:
        mask[n1:n2] = True

    indices = list(np.nonzero(mask)[0])

    return create_subset(stimuli, fixations, indices)


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


def filter_stimuli_by_size(stimuli, fixations, size=None, sizes=None):
    if size is not None:
        sizes = [size]
    sizes = [tuple(size) for size in sizes]

    indices = [i for i in range(len(stimuli)) if stimuli.sizes[i] in sizes]

    return create_subset(stimuli, fixations, indices)


def filter_scanpaths_by_attribute(scanpaths:  FixationTrains, attribute_name, attribute_value, invert_match=False):
    """Filter Scanpaths by values of scanpath attribute (fixation_trains.scanpath_attributes)"""
    
    mask = np.array([element == attribute_value for element in getattr(scanpaths, attribute_name)])
    if invert_match is True:
        mask = ~mask
    indices = list(np.nonzero(mask)[0])
    return scanpaths.filter_fixation_trains(indices)


def filter_fixations_by_attribute(fixations: Fixations, attribute_name, attribute_value, invert_match=False):
    """Filter Fixations by values of attribute (fixations.__attributes__)"""

    mask = np.array([element == attribute_value for element in getattr(fixations, attribute_name)])
    if invert_match is True:
        mask = ~mask
    indices = list(np.nonzero(mask)[0])
    return fixations.filter(indices)


def filter_stimuli_by_attribute(stimuli: Stimuli, fixations: Fixations, attribute_name, attribute_value, invert_match=False):
    """Filter stimuli by values of attribute"""

    mask = np.array([element == attribute_value for element in getattr(stimuli, attribute_name)])
    if invert_match is True:
        mask = ~mask
    indices = list(np.nonzero(mask)[0])
    return create_subset(stimuli, fixations, indices)


def filter_scanpaths_by_lengths(scanpaths: FixationTrains, intervals: list):
    """Filter Scanpaths by number of fixations"""
    
    intervals = _check_intervals(intervals, type=int)
    mask = np.zeros(len(scanpaths.train_lengths), dtype=bool)

    for n1, n2 in intervals:
        temp_mask = np.logical_and(scanpaths.train_lengths>=n1,scanpaths.train_lengths<=n2)
        mask = np.logical_or(mask, temp_mask)

    indices = list(np.nonzero(mask)[0])
    scanpaths = scanpaths.filter_fixation_trains(indices)

    return scanpaths
 
