from __future__ import absolute_import, print_function, division

import zipfile
import os

import requests
import numpy as np

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    download_and_check,
    atomic_directory_setup,
)

from .utils import create_stimuli, _load


def get_PASCAL_S(location=None):
    """
    Loads or downloads and caches the PASCAL-S dataset.
    The dataset consists of 850 images from the PASCAL-VOC
    2010 validation set with fixation from 12 subjects
    during 2s free-viewing.

    Note that here only the eye movement data from PASCAL-S
    is included. The original dataset also provides
    salient object segmentation data.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `PASCAL-S` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. seealso::

        Yin Li, Xiaodi Hu, Christof Koch, James M. Rehg, Alan L. Yuille:
        The Secrets of Salient Object Segmentation. CVPR 2014.

        http://cbs.ic.gatech.edu/salobj/
    """
    if location:
        location = os.path.join(location, 'PASCAL-S')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)

    n_stimuli = 850

    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:

            try:
                download_and_check('http://cbs.ic.gatech.edu/salobj/download/salObj.zip',
                                   os.path.join(temp_dir, 'salObj.zip'),
                                   'e48b4e5deac08bddcaec55ce56e4d420')
            except requests.exceptions.SSLError:
                print("http://cbs.ic.gatech.edu/salobj/download/salObj.zip seems to be using an invalid SSL certificate. Since this is known and since we're checking the MD5 sum in addition, we'll ignore the invalid certificate.")
                download_and_check('http://cbs.ic.gatech.edu/salobj/download/salObj.zip',
                    os.path.join(temp_dir, 'salObj.zip'),
                    'e48b4e5deac08bddcaec55ce56e4d420',
                    verify_ssl=False)

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'salObj.zip'))
            f.extractall(temp_dir)

            stimuli_src_location = os.path.join(temp_dir, 'datasets', 'imgs', 'pascal')
            stimuli_filenames = ['{}.jpg'.format(i + 1) for i in range(n_stimuli)]

            stimuli_target_location = os.path.join(location, 'Stimuli') if location else None
            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            print('Creating fixations')

            train_xs = []
            train_ys = []
            train_ts = []
            train_ns = []
            train_subjects = []

            import h5py  # we don't import globally to avoid depending on h5py
            with h5py.File(os.path.join(temp_dir, 'datasets', 'fixations', 'pascalFix.mat'), mode='r') as hdf5_file:
                fixation_data = [hdf5_file[hdf5_file['fixCell'][0, stimulus_index]][:] for stimulus_index in range(n_stimuli)]

            for n in range(n_stimuli):
                ys, xs, subject_ids = fixation_data[n]
                for subject in sorted(set(subject_ids)):
                    subject_inds = subject_ids == subject
                    if not np.any(subject_inds):
                        continue

                    train_xs.append(xs[subject_inds] - 1)  # data is 1-indexed in matlab
                    train_ys.append(ys[subject_inds] - 1)
                    train_ts.append(np.arange(subject_inds.sum()))
                    train_ns.append(n)
                    train_subjects.append(subject - 1)  # subjects are 1-indexed in matlab

        fixations = FixationTrains.from_fixation_trains(train_xs, train_ys, train_ts, train_ns, train_subjects)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations
