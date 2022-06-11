from __future__ import absolute_import, print_function, division

import zipfile
import os
import glob

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from ..datasets import Fixations, read_hdf5, clip_out_of_stimulus_fixations
from ..utils import (
    TemporaryDirectory,
    download_file_from_google_drive,
    check_file_hash,
    atomic_directory_setup)

from .utils import create_stimuli, _load



def get_SALICON(edition='2015', fixation_type='mouse', location=None):
    """
    Loads or downloads and caches the SALICON dataset as used in the LSUN challenge prior to 2017.
    For memory reasons no fixation trains  are provided.
    @type edition: string, defaults to '2015'
    @param edition: whether to provide the original SALICON dataset from 2015 or the update from 2017

    @type fixation_type: string, defaults to 'mouse'
    @param fixation_type: whether to use the mouse gaze postion ('mouse')
                          or the inferred fixations ('fixations').
                          For more details see the SALICON challenge homepage (below).

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `SALICON` of
                     location and read from there, if already present.
    @return: Training stimuli, validation stimuli, testing stimuli, training fixation trains, validation fixation trains

    .. seealso::
        Ming Jiang, Shengsheng Huang, Juanyong Duan*, Qi Zhao: SALICON: Saliency in Context, CVPR 2015

        http://salicon.net/

    .. note:
        prior to version 0.2.0 of pysaliency the data was stored in a way that allowed
        accessing the subject ids. This is not possible anymore and each scanpath on
        each image is just assigned an auto-incrementing id.
    """
    if edition not in ['2015', '2017']:
        raise ValueError('edition has to be \'2015\' or \'2017\', not \'{}\''.format(edition))

    if fixation_type not in ['mouse', 'fixations']:
        raise ValueError('fixation_type has to be \'mouse\' or \'fixations\', not \'{}\''.format(fixation_type))

    name = _get_SALICON_name(edition=edition, fixation_type=fixation_type)

    stimuli_train, stimuli_val, stimuli_test = _get_SALICON_stimuli(
        location=location, name='SALICON', edition=edition, fixation_type=fixation_type)
    fixations_train, fixations_val = _get_SALICON_fixations(
        location=location, name=name, edition=edition, fixation_type=fixation_type)

    return stimuli_train, stimuli_val, stimuli_test, fixations_train, fixations_val


def _get_SALICON_name(edition='2015', fixation_type='mouse'):
    if edition not in ['2015', '2017']:
        raise ValueError('edition has to be \'2015\' or \'2017\', not \'{}\''.format(edition))

    if fixation_type not in ['mouse', 'fixations']:
        raise ValueError('fixation_type has to be \'mouse\' or \'fixations\', not \'{}\''.format(fixation_type))

    if edition == '2015':
        if fixation_type == 'mouse':
            name = 'SALICON'
        elif fixation_type == 'fixations':
            name = 'SALICON2015_fixations'
    elif edition == '2017':
        if fixation_type == 'mouse':
            name = 'SALICON2017_mouse'
        elif fixation_type == 'fixations':
            name = 'SALICON2017_fixations'

    return name


def _get_SALICON_stimuli(location, name, edition='2015', fixation_type='mouse'):
    if edition not in ['2015', '2017']:
        raise ValueError('edition has to be \'2015\' or \'2017\', not \'{}\''.format(edition))

    if fixation_type not in ['mouse', 'fixations']:
        raise ValueError('fixation_type has to be \'mouse\' or \'fixations\', not \'{}\''.format(fixation_type))

    if location:
        location = os.path.join(location, name)
        if os.path.exists(location):
            if all(os.path.exists(os.path.join(location, filename)) for filename in ['stimuli_train.hdf5', 'stimuli_val.hdf5', 'stimuli_test.hdf5']):
                stimuli_train = read_hdf5(os.path.join(location, 'stimuli_train.hdf5'))
                stimuli_val = read_hdf5(os.path.join(location, 'stimuli_val.hdf5'))
                stimuli_test = read_hdf5(os.path.join(location, 'stimuli_test.hdf5'))
                return stimuli_train, stimuli_val, stimuli_test
        os.makedirs(location, exist_ok=True)

    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            stimuli_file = os.path.join(temp_dir, 'stimuli.zip')

            download_file_from_google_drive('1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5', stimuli_file)
            check_file_hash(stimuli_file, 'eb2a1bb706633d1b31fc2e01422c5757')

            print("Extracting stimuli")

            f = zipfile.ZipFile(stimuli_file)
            f.extractall(temp_dir)

            stimuli_train = create_stimuli(
                stimuli_location=os.path.join(temp_dir, 'images', 'train'),
                filenames=[os.path.basename(f) for f in sorted(glob.glob(os.path.join(temp_dir, 'images', 'train', 'COCO_train*')))],
                location=os.path.join(location, 'stimuli', 'train') if location else None
            )

            stimuli_val = create_stimuli(
                stimuli_location=os.path.join(temp_dir, 'images', 'val'),
                filenames=[os.path.basename(f) for f in sorted(glob.glob(os.path.join(temp_dir, 'images', 'val', 'COCO_val*')))],
                location=os.path.join(location, 'stimuli', 'val') if location else None
            )

            stimuli_test = create_stimuli(
                stimuli_location=os.path.join(temp_dir, 'images', 'test'),
                filenames=[os.path.basename(f) for f in sorted(glob.glob(os.path.join(temp_dir, 'images', 'test', 'COCO_test*')))],
                location=os.path.join(location, 'stimuli', 'test') if location else None
            )

            if location is not None:
                stimuli_train.to_hdf5(os.path.join(location, 'stimuli_train.hdf5'))
                stimuli_val.to_hdf5(os.path.join(location, 'stimuli_val.hdf5'))
                stimuli_test.to_hdf5(os.path.join(location, 'stimuli_test.hdf5'))

    return stimuli_train, stimuli_val, stimuli_test


def _get_SALICON_fixations(location, name, edition='2015', fixation_type='mouse'):
    if edition not in ['2015', '2017']:
        raise ValueError('edition has to be \'2015\' or \'2017\', not \'{}\''.format(edition))

    if fixation_type not in ['mouse', 'fixations']:
        raise ValueError('fixation_type has to be \'mouse\' or \'fixations\', not \'{}\''.format(fixation_type))

    if location:
        location = os.path.join(location, name)
        if os.path.exists(location):
            if all(os.path.exists(os.path.join(location, filename)) for filename in ['fixations_train.hdf5', 'fixations_val.hdf5']):
                fixations_train = read_hdf5(os.path.join(location, 'fixations_train.hdf5'))
                fixations_val = read_hdf5(os.path.join(location, 'fixations_val.hdf5'))
                return fixations_train, fixations_val
        os.makedirs(location, exist_ok=True)

    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            fixations_file = os.path.join(temp_dir, 'fixations.zip')

            if edition == '2015':
                download_file_from_google_drive('1WVEiXba-I4GN33f0uUl4KhaN1rK7qzs6', fixations_file)
                check_file_hash(fixations_file, '9a22db9d718200fb90252e5010c004c4')
            elif edition == '2017':
                download_file_from_google_drive('1P-jeZXCsjoKO79OhFUgnj6FGcyvmLDPj', fixations_file)
                check_file_hash(fixations_file, '462b70f4f9e8ea446ac628e46cea8d3d')

            f = zipfile.ZipFile(fixations_file)
            f.extractall(os.path.join(temp_dir, 'fixations'))

            fixations = []

            if fixation_type == 'mouse':
                fixation_attr = 'location'
            elif fixation_type == 'fixations':
                fixation_attr = 'fixations'

            for dataset in ['train', 'val']:
                ns = []
                train_xs = []
                train_ys = []
                train_ts = []
                train_subjects = []

                subject_id = 0

                data_files = list(sorted(glob.glob(os.path.join(temp_dir, 'fixations', dataset, '*.mat'))))
                for n, filename in enumerate(tqdm(data_files)):
                    fixation_data = loadmat(filename)
                    for subject_data in fixation_data['gaze'].flatten():
                        train_xs.append(subject_data[fixation_attr][:, 0] - 1)  # matlab: one-based indexing
                        train_ys.append(subject_data[fixation_attr][:, 1] - 1)
                        if fixation_type == 'mouse':
                            train_ts.append(subject_data['timestamp'].flatten())
                        elif fixation_type == 'fixations':
                            train_ts.append(range(len(train_xs[-1])))

                        train_subjects.append(np.ones(len(train_xs[-1]), dtype=int) * subject_id)
                        ns.append(np.ones(len(train_xs[-1]), dtype=int) * n)
                        subject_id += 1

                xs = np.hstack(train_xs)
                ys = np.hstack(train_ys)
                ts = np.hstack(train_ts)
                subjects = np.hstack(train_subjects)
                ns = np.hstack(ns)

                fixations.append(Fixations.FixationsWithoutHistory(xs, ys, ts, ns, subjects))

            fixations_train, fixations_val = fixations

            if edition == '2017' and fixation_type == 'mouse':
                fixations_train = clip_out_of_stimulus_fixations(fixations_train, width=640, height=480)
                fixations_val = clip_out_of_stimulus_fixations(fixations_val, width=640, height=480)

            if location is not None:
                fixations_train.to_hdf5(os.path.join(location, 'fixations_train.hdf5'))
                fixations_val.to_hdf5(os.path.join(location, 'fixations_val.hdf5'))

        return fixations_train, fixations_val


def get_SALICON_train(edition='2015', fixation_type='mouse', location=None):
    """
    Loads or downloads and caches the SALICON training dataset. See `get_SALICON` for more details.
    """
    if location:
        name = _get_SALICON_name(edition=edition, fixation_type=fixation_type)
        if os.path.exists(os.path.join(location, name)):
            stimuli = _load(os.path.join(location, 'SALICON', 'stimuli_train.hdf5'))
            fixations = _load(os.path.join(location, name, 'fixations_train.hdf5'))
            return stimuli, fixations
    stimuli_train, _, _, fixations_train, _ = get_SALICON(location=location, edition=edition, fixation_type=fixation_type)

    return stimuli_train, fixations_train


def get_SALICON_val(edition='2015', fixation_type='mouse', location=None):
    """
    Loads or downloads and caches the SALICON validation dataset. See `get_SALICON` for more details.
    """
    if location:
        name = _get_SALICON_name(edition=edition, fixation_type=fixation_type)
        if os.path.exists(os.path.join(location, name)):
            stimuli = _load(os.path.join(location, 'SALICON', 'stimuli_val.hdf5'))
            fixations = _load(os.path.join(location, name, 'fixations_val.hdf5'))
            return stimuli, fixations
    _, stimuli_val, _, _, fixations_val = get_SALICON(location=location, edition=edition, fixation_type=fixation_type)

    return stimuli_val, fixations_val


def get_SALICON_test(edition='2015', fixation_type='mouse', location=None):
    """
    Loads or downloads and caches the SALICON test dataset. See `get_SALICON` for more details.
    """
    if location:
        name = _get_SALICON_name(edition=edition, fixation_type=fixation_type)
        if os.path.exists(os.path.join(location, name)):
            stimuli = _load(os.path.join('SALICON', 'stimuli_test.hdf5'))
            return stimuli
    _, _, stimuli_test, _, _ = get_SALICON(location=location, edition=edition, fixation_type=fixation_type)

    return stimuli_test