from __future__ import absolute_import, print_function, division

import os
import warnings
import zipfile


import numpy as np
from scipy.io import loadmat

from ..datasets import FixationTrains, Fixations
from ..utils import (
    TemporaryDirectory,
    download_and_check,
    atomic_directory_setup)

from .utils import create_stimuli, _load


def get_toronto(location=None):
    """
    Loads or downloads and caches the Toronto dataset. The dataset
    consists of 120 color images of outdoor and indoor scenes
    of size 681x511px and the fixations of 20 subjects under
    free viewing conditions with 4 seconds presentation time.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, Fixations

    .. warning::
        At the moment, the subjects stated in the FixationTrains object
        will not be correct, as they are difficult to infer from the
        published data (the data per subject is not stated in image dimensions)

    .. seealso::

        Neil Bruce, John K. Tsotsos. Attention based on information maximization [JoV 2007]

        `http://www-sop.inria.fr/members/Neil.Bruce/#SOURCECODE
    """
    if location:
        location = os.path.join(location, 'toronto')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory() as temp_dir:
            src = 'http://www-sop.inria.fr/members/Neil.Bruce/eyetrackingdata.zip'
            target = os.path.join(temp_dir, 'eyetrackingdata.zip')
            md5_sum = '38d5c02217060d4d2d1a4649cc632af1'
            download_and_check(src, target, md5_sum)
            z = zipfile.ZipFile(target)
            print('Extracting')
            z.extractall(temp_dir)

            # Stimuli
            stimuli_src_location = os.path.join(temp_dir, 'eyetrackingdata', 'fixdens', 'Original Image Set')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            stimuli_filenames = ['{}.jpg'.format(i) for i in range(1, 121)]

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            points = loadmat(os.path.join(temp_dir, 'eyetrackingdata', 'fixdens', 'origfixdata.mat'))['white']
            xs = []
            ys = []
            ns = []
            subjects = []
            for n in range(len(stimuli.stimuli)):
                _ys, _xs = np.nonzero(points[0, n])
                xs.extend(_xs)
                ys.extend(_ys)
                ns.extend([n for x in _xs])
                subjects.extend([0 for x in _xs])
            fixations = Fixations.create_without_history(xs, ys, ns, subjects=subjects)
        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


def get_toronto_with_subjects(location=None):
    """
    Loads or downloads and caches the Toronto dataset. The dataset
    consists of 120 color images of outdoor and indoor scenes
    of size 681x511px and the fixations of 20 subjects under
    free viewing conditions with 4 seconds presentation time.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. warning::
        This function uses the positions as given per subject in the toronto
        dataset. Unfortunately, these positions do not seem to be pixel positions
        as the fixations are not located as in the file `origfixdata.mat'.
        The code is still in the package as a template for fixing this issue.

    .. seealso::

        Neil Bruce, John K. Tsotsos. Attention based on information maximization [JoV 2007]

        `http://www-sop.inria.fr/members/Neil.Bruce/#SOURCECODE
    """
    warnings.warn('This function reports wrong fixation positions! See docstring for details.')
    if location:
        location = os.path.join(location, 'toronto')
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory() as temp_dir:
            src = 'http://www-sop.inria.fr/members/Neil.Bruce/eyetrackingdata.zip'
            target = os.path.join(temp_dir, 'eyetrackingdata.zip')
            md5_sum = '38d5c02217060d4d2d1a4649cc632af1'
            download_and_check(src, target, md5_sum)
            z = zipfile.ZipFile(target)
            print('Extracting')
            z.extractall(temp_dir)

            # Stimuli
            stimuli_src_location = os.path.join(temp_dir, 'eyetrackingdata', 'fixdens', 'Original Image Set')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            stimuli_filenames = ['{}.jpg'.format(i) for i in range(1, 121)]

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            print("Getting fixations")
            raw_path = os.path.join(temp_dir, 'eyetrackingdata', 'fixdens', 'Raw')
            subjects = os.listdir(raw_path)
            train_xs = []
            train_ys = []
            train_ts = []
            train_ns = []
            train_subjects = []
            subjects = [s for s in subjects if s != 'processed']
            subjects = sorted(subjects)
            for subject_nr, subject in enumerate(subjects):
                print("Doing subject", subject)
                for n, image in enumerate(stimuli_filenames):
                    imagename = os.path.splitext(os.path.split(image)[-1])[0]
                    with open(os.path.join(raw_path, subject, imagename + '.fix')) as f:
                        content = f.read()
                    content = content.replace('\r', '')
                    if 'No fixations' in content:
                        print("No fixations for {}, skipping".format(image))
                        continue
                    subject_fixations = content.split('Fixation Listing')[1].split('\n\n')[1].split('Average')[0]
                    _xs = []
                    _ys = []
                    _ts = []
                    for line in subject_fixations.split('\n'):
                        if not line:
                            continue
                        parts = line.split()
                        parts = [p.replace(',', '') for p in parts]
                        _xs.append(int(parts[1]))
                        _ys.append(int(parts[2]))
                        _ts.append(float(parts[3]))
                    _xs = np.array(_xs, dtype=np.float)
                    _ys = np.array(_ys, dtype=np.float)
                    _ts = np.array(_ts, dtype=np.float)
                    xs = []
                    ys = []
                    ts = []
                    for i in range(len(_xs)):
                        if _xs[i] > 680:
                            continue
                        if _ys[i] > 510:
                            continue
                        xs.append(_xs[i])
                        ys.append(_ys[i])
                        ts.append(_ts[i])
                    train_xs.append(xs)
                    train_ys.append(ys)
                    train_ts.append(ts)
                    train_ns.append(n)
                    train_subjects.append(subject_nr)

            fixations = FixationTrains.from_fixation_trains(train_xs, train_ys, train_ts, train_ns, train_subjects)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations
