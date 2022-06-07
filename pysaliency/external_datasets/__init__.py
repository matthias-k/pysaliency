from __future__ import absolute_import, print_function, division

import zipfile
import os
import glob
from six.moves import urllib

import numpy as np
from scipy.io import loadmat
from natsort import natsorted
from boltons.fileutils import mkdir_p
from tqdm import tqdm

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    download_and_check,
    atomic_directory_setup,
)

from .utils import create_stimuli, _load

from .toronto import get_toronto, get_toronto_with_subjects
from .mit import get_mit1003, get_mit1003_with_initial_fixation, get_mit1003_onesize, get_mit300
from .cat2000 import get_cat2000_test, get_cat2000_train
from .isun import get_iSUN, get_iSUN_training, get_iSUN_validation, get_iSUN_testing
from .salicon import get_SALICON, get_SALICON_train, get_SALICON_val, get_SALICON_test
from .koehler import get_koehler
from .figrim import get_FIGRIM
from .osie import get_OSIE
from .nusef import get_NUSEF_public


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

            download_and_check('http://cbs.ic.gatech.edu/salobj/download/salObj.zip',
                               os.path.join(temp_dir, 'salObj.zip'),
                               'e48b4e5deac08bddcaec55ce56e4d420')

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


def get_DUT_OMRON(location=None):
    """
    Loads or downloads the DUT-OMRON fixation dataset.
    The dataset consists of 5168 natural images with
    a maximal size of 400 pixel and eye movement data
    from 5 subjects in a 2 second free viewing task.
    The eye movement data has been filtered and preprocessed,
    see the dataset documentation for more details.

    Note that the dataset contains subject ids but they
    might not be consisten across images.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `DUT-OMRON` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. seealso::

        Chuan Yang, Lihe Zhang, Huchuan Lu, Xiang Ruan, Minghsuan Yang. Saliency Detection Via Graph-Based Manifold Ranking, CVPR2013.

        http://saliencydetection.net/dut-omron
    """
    if location:
        location = os.path.join(location, 'DUT-OMRON')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)

    n_fixations = 0

    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:

            download_and_check('http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip',
                               os.path.join(temp_dir, 'DUT-OMRON-image.zip'),
                               'a8951db9297afacf78bc0e5079103cf1')

            download_and_check('http://saliencydetection.net/dut-omron/download/DUT-OMRON-eye-fixations.zip',
                               os.path.join(temp_dir, 'DUT-OMRON-eye-fixations.zip'),
                               'd9f4f83fcc78b1e5efb579ae9fb0edc2')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'DUT-OMRON-image.zip'))
            f.extractall(temp_dir)

            stimuli_src_location = os.path.join(temp_dir, 'DUT-OMRON-image')
            images = glob.glob(os.path.join(stimuli_src_location, '*.jpg'))
            images = [os.path.relpath(img, start=stimuli_src_location) for img in images]
            stimuli_filenames = sorted(images)

            stimuli_target_location = os.path.join(location, 'Stimuli') if location else None
            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            stimuli_basenames = [os.path.basename(f) for f in stimuli_filenames]

            # FixationTrains

            print('Creating fixations')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'DUT-OMRON-eye-fixations.zip'))
            f.extractall(temp_dir)

            train_xs = []
            train_ys = []
            train_ts = []
            train_ns = []
            train_subjects = []

            for n, basename in enumerate(tqdm(stimuli_basenames)):
                eye_filename = os.path.join(temp_dir, 'DUT-OMRON-eye-fixations', 'mat', basename.replace('.jpg', '.mat'))
                eye_data = loadmat(eye_filename)['s']
                xs, ys, subject_ids = eye_data.T
                n_fixations += len(xs) - 1  # first entry is image size
                for subject_index in range(subject_ids.max()):
                    subject_inds = subject_ids == subject_index + 1  # subject==0 is image size

                    if not np.any(subject_inds):
                        continue

                    # since there are coordinates with value 0, we assume they are 0-indexed (although they are matlab)
                    train_xs.append(xs[subject_inds])
                    train_ys.append(ys[subject_inds])
                    train_ts.append(np.arange(subject_inds.sum()))
                    train_ns.append(n)
                    train_subjects.append(subject_index)

        fixations = FixationTrains.from_fixation_trains(train_xs, train_ys, train_ts, train_ns, train_subjects)

        assert len(fixations) == n_fixations

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations
