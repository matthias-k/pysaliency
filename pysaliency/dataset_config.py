from .datasets import read_hdf5, clip_out_of_stimulus_fixations, remove_out_of_stimulus_fixations
from .filter_datasets import (
    filter_fixations_by_number,
    filter_stimuli_by_number,
    filter_stimuli_by_size,
    train_split,
    validation_split,
    test_split,
    filter_scanpaths_by_attribute,
    filter_fixations_by_attribute,
    filter_stimuli_by_attribute,
    filter_scanpaths_by_length,
    remove_stimuli_without_fixations
)

from schema import Schema, Optional, Or


dataset_config_schema = Schema({
    'stimuli': str,
    'fixations': str,
    Optional('filters', default=[]): [{
        'type': str,
        Optional('parameters', default={}): dict,
    }],
    Optional('lmdb_path', default=None): Or(str, None),
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
        'filter_scanpaths_by_attribute': add_stimuli_argument(filter_scanpaths_by_attribute),
        'filter_fixations_by_attribute': add_stimuli_argument(filter_fixations_by_attribute),
        'filter_stimuli_by_attribute': filter_stimuli_by_attribute,
        'filter_scanpaths_by_length': add_stimuli_argument(filter_scanpaths_by_length),
        'remove_stimuli_without_fixations': remove_stimuli_without_fixations
    }

    if filter_config['type'] not in filter_dict:
        raise ValueError("Invalid filter name: {}".format(filter_config['type']))

    filter_fn = filter_dict[filter_config['type']]

    return filter_fn(stimuli, fixations, **filter_config['parameters'])


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
