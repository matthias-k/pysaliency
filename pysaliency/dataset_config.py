from .datasets import read_hdf5
from .filter_datasets import filter_fixations_by_number

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
    if filter_config['type'] == 'filter_fixations_by_number':
        filter_fn = add_stimuli_argument(filter_fixations_by_number)

    return filter_fn(stimuli, fixations, **filter_config['parameters'])


def add_stimuli_argument(fn):
    def wrapped(stimuli, fixations, **kwargs):
        new_fixations = fn(fixations, **kwargs)
        return stimuli, new_fixations

    return wrapped
