import pathlib
from typing import Dict, List, Optional, Union
from weakref import WeakValueDictionary

import numpy as np
from boltons.cacheutils import cached

from .fixations import Fixations, FixationTrains, scanpaths_from_fixations
from .scanpaths import Scanpaths
from .stimuli import FileStimuli, ObjectStimuli, Stimuli, StimuliStimulus, Stimulus, as_stimulus, check_prediction_shape, get_image_hash
from .utils import _load_attribute_dict_from_hdf5, concatenate_attributes, create_hdf5_dataset, decode_string, get_merged_attribute_list, hdf5_wrapper


@cached(WeakValueDictionary())
def _read_hdf5_from_file(source):
    import h5py
    with h5py.File(source, 'r') as hdf5_file:
        return read_hdf5(hdf5_file)


def read_hdf5(source):
    if isinstance(source, (str, pathlib.Path)):
        return _read_hdf5_from_file(source)

    data_type = decode_string(source.attrs['type'])

    if data_type == 'Fixations':
        return Fixations.read_hdf5(source)
    elif data_type == 'FixationTrains':
        return FixationTrains.read_hdf5(source)
    elif data_type == 'Scanpaths':
        return Scanpaths.read_hdf5(source)
    elif data_type == 'Stimuli':
        return Stimuli.read_hdf5(source)
    elif data_type == 'FileStimuli':
        return FileStimuli.read_hdf5(source)
    else:
        raise ValueError("Invalid HDF content type:", data_type)


def create_subset(stimuli, fixations, stimuli_indices):
    """Create subset of stimuli and fixations using only stimuli
    with given indices.
    """
    if isinstance(stimuli_indices, np.ndarray) and stimuli_indices.dtype == bool:
        if len(stimuli_indices) != len(stimuli):
            raise ValueError("length of mask doesn't match stimuli")
        stimuli_indices = np.nonzero(stimuli_indices)[0]

    new_stimuli = stimuli[stimuli_indices]
    if isinstance(fixations, FixationTrains):
        fix_inds = np.in1d(fixations.scanpaths.n, stimuli_indices)

        index_list = list(stimuli_indices)
        new_pos = {i: index_list.index(i) for i in index_list}

        new_image_indices = [new_pos[i] for i in fixations.scanpaths.n[fix_inds]]

        new_scanpaths = fixations.scanpaths[fix_inds]

        new_fixations = FixationTrains(
            train_xs=new_scanpaths.xs,
            train_ys=new_scanpaths.ys,
            train_ts=None, # new_scanpaths.fixation_attributes['ts'],
            train_ns=np.array(new_image_indices),
            train_subjects=None, #  new_scanpaths.scanpath_attributes['subject'],
            scanpath_attributes=new_scanpaths.scanpath_attributes,
            scanpath_fixation_attributes=new_scanpaths.fixation_attributes,
            scanpath_attribute_mapping=new_scanpaths.attribute_mapping
        )

    else:
        fix_inds = np.in1d(fixations.n, stimuli_indices)
        new_fixations = fixations[fix_inds]

        index_list = list(stimuli_indices)
        new_pos = {i: index_list.index(i) for i in index_list}
        new_fixation_ns = [new_pos[i] for i in new_fixations.n]
        new_fixations.n = np.array(new_fixation_ns)

    return new_stimuli, new_fixations


def concatenate_stimuli(stimuli):
    attributes = {}
    for key in get_merged_attribute_list([list(s.attributes.keys()) for s in stimuli]):
        attributes[key] = concatenate_attributes(s.attributes[key] for s in stimuli)

    if all(isinstance(s, FileStimuli) for s in stimuli):
        return FileStimuli(sum([s.filenames for s in stimuli], []), attributes=attributes)
    else:
        return ObjectStimuli(sum([s.stimulus_objects for s in stimuli], []), attributes=attributes)


#np.testing.assert_allclose(concatenate_attributes([[0], [1, 2, 3]]), [0,1,2,3])
#np.testing.assert_allclose(concatenate_attributes([[[0]], [[1],[2], [3]]]), [[0],[1],[2],[3]])
#np.testing.assert_allclose(concatenate_attributes([[[0.,1.]], [[1.],[2.], [3.]]]), [[0, 1],[1,np.nan],[2,np.nan],[3,np.nan]])


def concatenate_fixations(fixations):
    if all(isinstance(f, FixationTrains) for f in fixations):
        return FixationTrains.concatenate(fixations)
    else:
        return Fixations.concatenate(fixations)


def concatenate_datasets(stimuli, fixations):
    """Concatenate multiple Stimuli instances with associated fixations"""

    stimuli = list(stimuli)
    fixations = list(fixations)
    assert len(stimuli) == len(fixations)
    if len(stimuli) == 1:
        return stimuli[0], fixations[0]

    for i in range(len(fixations)):
        offset = sum(len(s) for s in stimuli[:i])
        f = fixations[i].copy()
        f.n += offset
        if isinstance(f, FixationTrains):
            f.train_ns += offset
        fixations[i] = f

    return concatenate_stimuli(stimuli), concatenate_fixations(fixations)


def remove_out_of_stimulus_fixations(stimuli, fixations):
    """ Return all fixations which do not occour outside the stimulus
    """
    widths = np.array([s[1] for s in stimuli.sizes])
    heights = np.array([s[0] for s in stimuli.sizes])

    inds = ((fixations.x >= 0) & (fixations.y >= 0) &
            (fixations.x < widths[fixations.n]) &
            (fixations.y < heights[fixations.n])
            )
    return fixations[inds]


def clip_out_of_stimulus_fixations(fixations, stimuli=None, width=None, height=None):
    if stimuli is None and (width is None or height is None):
        raise ValueError("You have to provide either stimuli or width and height")
    if stimuli is not None:
        widths = np.array([s[1] for s in stimuli.sizes])
        heights = np.array([s[0] for s in stimuli.sizes])
        new_fixations = fixations.copy()
        x_max = widths[fixations.n] - 0.01
        y_max = heights[fixations.n] - 0.01
    else:
        x_max = width - 0.01
        y_max = height - 0.01

    new_fixations = fixations.copy()

    new_fixations.x = np.clip(new_fixations.x, a_min=0, a_max=x_max)
    new_fixations.y = np.clip(new_fixations.y, a_min=0, a_max=y_max)

    if isinstance(x_max, np.ndarray):
        x_max = x_max[:, np.newaxis]
        y_max = y_max[:, np.newaxis]

    new_fixations.x_hist = np.clip(new_fixations.x_hist, a_min=0, a_max=x_max)
    new_fixations.y_hist = np.clip(new_fixations.y_hist, a_min=0, a_max=y_max)

    if isinstance(fixations, FixationTrains):
        if stimuli is not None:
            x_max = widths[fixations.train_ns] - 0.01
            y_max = heights[fixations.train_ns] - 0.01
            x_max = x_max[:, np.newaxis]
            y_max = y_max[:, np.newaxis]
        else:
            x_max = width - 0.01
            y_max = height - 0.01

        x_max = widths[fixations.train_ns] - 0.01
        y_max = heights[fixations.train_ns] - 0.01
        new_fixations.train_xs = np.clip(new_fixations.train_xs, a_min=0, a_max=x_max[:, np.newaxis])
        new_fixations.train_ys = np.clip(new_fixations.train_ys, a_min=0, a_max=y_max[:, np.newaxis])

    return new_fixations


def calculate_nonfixation_factors(stimuli, index):
    widths = np.asarray([s[1] for s in stimuli.sizes]).astype(float)
    heights = np.asarray([s[0] for s in stimuli.sizes]).astype(float)

    x_factors = stimuli.sizes[index][1] / widths
    y_factors = stimuli.sizes[index][0] / heights

    return x_factors, y_factors


def create_nonfixations(stimuli, fixations, index, adjust_n = True, adjust_history=True):
    """Create nonfixations from fixations for given index

    stimuli of different sizes will be rescaled to match the
    target stimulus
    """

    x_factors, y_factors = calculate_nonfixation_factors(stimuli, index)

    non_fixations = fixations[fixations.n != index]
    other_ns = non_fixations.n

    non_fixations.x = non_fixations.x * x_factors[other_ns]
    non_fixations.y = non_fixations.y * y_factors[other_ns]

    if adjust_history:
        non_fixations.x_hist = non_fixations.x_hist * x_factors[other_ns][:, np.newaxis]
        non_fixations.y_hist = non_fixations.y_hist * y_factors[other_ns][:, np.newaxis]

    if adjust_n:
        non_fixations.n = np.ones(len(non_fixations.n), dtype=int)*index

    return non_fixations