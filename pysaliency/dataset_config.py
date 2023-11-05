import numpy as np
from .datasets import read_hdf5, clip_out_of_stimulus_fixations, remove_out_of_stimulus_fixations, FixationTrains, Fixations, Stimuli, create_subset
from .filter_datasets import (
    filter_fixations_by_number,
    filter_stimuli_by_number,
    filter_stimuli_by_size,
    train_split,
    validation_split,
    test_split,
    _check_intervals
)

from schema import Schema, Optional


dataset_config_schema = Schema({
    'stimuli': str,
    'fixations': str,
    Optional('filters', default=[]): [{
        'type': str,
        Optional('parameters', default={}): dict,
    }],
})


def load_dataset_from_config(config):
    config = dataset_config_schema.validate(config)
    stimuli = read_hdf5(config['stimuli'])
    fixations = read_hdf5(config['fixations'])

    for filter_config in config['filters']:
        stimuli, fixations = apply_dataset_filter_config(stimuli, fixations, filter_config)

    return stimuli, fixations


def apply_dataset_filter_config(stimuli, fixations, filter_config):
    filter_dict = {
        'filter_fixations_by_number': add_stimuli_argument(filter_fixations_by_number),
        'filter_stimuli_by_number': filter_stimuli_by_number,
        'filter_stimuli_by_size': filter_stimuli_by_size,
        'clip_out_of_stimulus_fixations': _clip_out_of_stimulus_fixations,
        'remove_out_of_stimulus_fixations': _remove_out_of_stimulus_fixations,
        'train_split': train_split,
        'validation_split': validation_split,
        'test_split': test_split,
    }

    if filter_config['type'] not in filter_dict:
        raise ValueError("Invalid filter name: {}".format(filter_config['type']))

    filter_fn = filter_dict[filter_config['type']]

    return filter_fn(stimuli, fixations, **filter_config['parameters'])


def filter_scanpaths_by_attribute(scanpaths: FixationTrains, whitelist: dict=None, blacklist: dict=None):
    """Filter Scanpaths by values of scanpath attributes (fixation_trains.scanpath_attributes), the dictionary can have only one attribute"""
    
    assert (whitelist is None and blacklist is not None) or (whitelist is not None and blacklist is None)
    if whitelist is not None:
        assert(len(whitelist)==1)
    if blacklist is not None:
        assert(len(blacklist)==1)

    if whitelist is not None:
        attribute_name = list(whitelist.keys())[0]
        attribute_value = list(whitelist.values())[0]

        mask = np.zeros(len(getattr(scanpaths, attribute_name)), dtype=bool)

        mask = np.logical_or(mask,[element == attribute_value for element in getattr(scanpaths, attribute_name)])
        indices = list(np.nonzero(mask)[0])
        return scanpaths.filter_fixation_trains(indices)
    
    if blacklist is not None:
        attribute_name = list(blacklist.keys())[0]
        attribute_value = list(blacklist.values())[0]

        mask = np.zeros(len(getattr(scanpaths, attribute_name)), dtype=bool)

        mask = np.logical_or(mask,[element == attribute_value for element in getattr(scanpaths, attribute_name)])
        mask = ~mask
        indices = list(np.nonzero(mask)[0])
        return scanpaths.filter_fixation_trains(indices)


def filter_fixations_by_attribute(fixations: Fixations, whitelist: dict=None, blacklist: dict=None):
    """Filter Fixations by values of attributes (fixations.__attributes__), the dictionary can have only one attribute"""
    
    assert (whitelist is None and blacklist is not None) or (whitelist is not None and blacklist is None)
    if whitelist is not None:
        assert(len(whitelist)==1)
    if blacklist is not None:
        assert(len(blacklist)==1)
    
    if whitelist is not None:
        attribute_name = list(whitelist.keys())[0]
        attribute_value = list(whitelist.values())[0]

        mask = np.zeros(len(getattr(fixations, attribute_name)), dtype=bool)

        mask = np.logical_or(mask,[element == attribute_value for element in getattr(fixations, attribute_name)])
        indices = list(np.nonzero(mask)[0])
        return fixations.filter(indices)
    
    if blacklist is not None:
        attribute_name = list(blacklist.keys())[0]
        attribute_value = list(blacklist.values())[0]

        mask = np.zeros(len(getattr(fixations, attribute_name)), dtype=bool)

        mask = np.logical_or(mask,[element == attribute_value for element in getattr(fixations, attribute_name)])
        mask = ~mask
        indices = list(np.nonzero(mask)[0])
        return fixations.filter(indices)


def filter_stimuli_by_attribute(stimuli: Stimuli, fixations: Fixations, whitelist: dict=None, blacklist: dict=None):
    """Filter stimuli by values of attribute, the dictionary can have only one attribute"""
    
    assert (whitelist is None and blacklist is not None) or (whitelist is not None and blacklist is None)
    if whitelist is not None:
        assert(len(whitelist)==1)
    if blacklist is not None:
        assert(len(blacklist)==1)

    if whitelist is not None:
        attribute_name = list(whitelist.keys())[0]
        attribute_value = list(whitelist.values())[0]

        mask = np.zeros(len(stimuli), dtype=bool)

        mask = np.logical_or(mask,[element == attribute_value for element in getattr(stimuli, attribute_name)])
        indices = list(np.nonzero(mask)[0])
        return create_subset(stimuli, fixations, indices)
    
    if blacklist is not None:
        attribute_name = list(blacklist.keys())[0]
        attribute_value = list(blacklist.values())[0]

        mask = np.zeros(len(stimuli), dtype=bool)

        mask = np.logical_or(mask,[element == attribute_value for element in getattr(stimuli, attribute_name)])
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

    
def _clip_out_of_stimulus_fixations(stimuli, fixations):
    clipped_fixations = clip_out_of_stimulus_fixations(fixations, stimuli=stimuli)
    return stimuli, clipped_fixations


def _remove_out_of_stimulus_fixations(stimuli, fixations):
    filtered_fixations = remove_out_of_stimulus_fixations(stimuli, fixations)
    return stimuli, filtered_fixations


def add_stimuli_argument(fn):
    def wrapped(stimuli, fixations, **kwargs):
        new_fixations = fn(fixations, **kwargs)
        return stimuli, new_fixations

    return wrapped
