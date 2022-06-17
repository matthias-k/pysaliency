from __future__ import absolute_import, print_function, division

import zipfile
import os

from scipy.io import loadmat

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    filter_files,
    download_and_check,
    atomic_directory_setup)

from .utils import create_stimuli, _load


# TODO: Add test
def get_iSUN(location=None):
    """
    Loads or downloads and caches the iSUN dataset.
    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `iSUN` of
                     location and read from there, if already present.
    @return: Training stimuli, validation stimuli, testing stimuli, training fixation trains, validation fixation trains

    .. seealso::

        P. Xu, K. A. Ehinger, Y. Zhang, A. Finkelstein, S. R. Kulkarni, and J. Xiao.: TurkerGaze: Crowdsourcing Saliency with Webcam based Eye Tracking

        http://lsun.cs.princeton.edu/

        http://vision.princeton.edu/projects/2014/iSUN/
    """
    if location:
        location = os.path.join(location, 'iSUN')
        if os.path.exists(location):
            stimuli_training = _load(os.path.join(location, 'stimuli_training.hdf5'))
            stimuli_validation = _load(os.path.join(location, 'stimuli_validation.hdf5'))
            stimuli_testing = _load(os.path.join(location, 'stimuli_testing.hdf5'))
            fixations_training = _load(os.path.join(location, 'fixations_training.hdf5'))
            fixations_validation = _load(os.path.join(location, 'fixations_validation.hdf5'))
            return stimuli_training, stimuli_validation, stimuli_testing, fixations_training, fixations_validation
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://lsun.cs.princeton.edu/challenge/2015/eyetracking/data/training.mat',
                               os.path.join(temp_dir, 'training.mat'),
                               '5a8b15134b17c7a3f69b087845db1363')
            download_and_check('http://lsun.cs.princeton.edu/challenge/2015/eyetracking/data/validation.mat',
                               os.path.join(temp_dir, 'validation.mat'),
                               'f68e9b011576e48d2460b883854fd86c')
            download_and_check('http://lsun.cs.princeton.edu/challenge/2015/eyetracking/data/testing.mat',
                               os.path.join(temp_dir, 'testing.mat'),
                               'be008ef0330467dcb9c9cd9cc96a8546')
            download_and_check('http://lsun.cs.princeton.edu/challenge/2015/eyetracking/data/fixation.zip',
                               os.path.join(temp_dir, 'fixation.zip'),
                               'aadc15784e1b0023cda4536335b7839c')
            download_and_check('http://lsun.cs.princeton.edu/challenge/2015/eyetracking/data/image.zip',
                               os.path.join(temp_dir, 'image.zip'),
                               '0a3af01c5307f1d44f5dd309f71ea963')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'image.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.DS_Store'])
            f.extractall(temp_dir, namelist)

            def get_stimuli_names(name):
                data_file = os.path.join(temp_dir, '{}.mat'.format(name))
                data = loadmat(data_file)[name]
                stimuli_names = [d[0] for d in data['image'][:, 0]]
                stimuli_names = ['{}.jpg'.format(n) for n in stimuli_names]
                return stimuli_names

            stimulis = []
            stimuli_src_location = os.path.join(temp_dir, 'images')
            for name in ['training', 'validation', 'testing']:
                print("Creating {} stimuli".format(name))
                stimuli_target_location = os.path.join(location, 'stimuli_{}'.format(name)) if location else None
                images = get_stimuli_names(name)
                stimulis.append(create_stimuli(stimuli_src_location, images, stimuli_target_location))

            # FixationTrains
            print('Creating fixations')

            def get_fixations(name):
                data_file = os.path.join(temp_dir, '{}.mat'.format(name))
                data = loadmat(data_file)[name]
                gaze = data['gaze'][:, 0]
                ns = []
                train_xs = []
                train_ys = []
                train_ts = []
                train_subjects = []
                for n in range(len(gaze)):
                    fixation_trains = gaze[n]['fixation'][0, :]
                    for train in fixation_trains:
                        xs = train[:, 0]
                        ys = train[:, 1]
                        ns.append(n)
                        train_xs.append(xs)
                        train_ys.append(ys)
                        train_ts.append(range(len(xs)))
                        train_subjects.append(0)
                fixations = FixationTrains.from_fixation_trains(train_xs, train_ys, train_ts, ns, train_subjects)
                return fixations

            fixations = []
            for name in ['training', 'validation']:
                print("Creating {} fixations".format(name))
                fixations.append(get_fixations(name))

        if location:
            stimulis[0].to_hdf5(os.path.join(location, 'stimuli_training.hdf5'))
            stimulis[1].to_hdf5(os.path.join(location, 'stimuli_validation.hdf5'))
            stimulis[2].to_hdf5(os.path.join(location, 'stimuli_test.hdf5'))
            fixations[0].to_hdf5(os.path.join(location, 'fixations_training.hdf5'))
            fixations[1].to_hdf5(os.path.join(location, 'fixations_validation.hdf5'))

    return stimulis + fixations


def get_iSUN_training(location=None):
    """
    @return: Training stimuli, training fixation trains

    See `get_iSUN` for more information"""
    training_stimuli, _, _, training_fixations, _ = get_iSUN(location=location)
    return training_stimuli, training_fixations


def get_iSUN_validation(location=None):
    """
    @return: validation stimuli, validation fixation trains

    See `get_iSUN` for more information"""
    _, validation_stimuli, _, _, validation_fixations = get_iSUN(location=location)
    return validation_stimuli, validation_fixations


def get_iSUN_testing(location=None):
    """
    @return: testing stimuli

    See `get_iSUN` for more information"""
    _, _, test_stimuli, _, _ = get_iSUN(location=location)
    return test_stimuli