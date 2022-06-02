from __future__ import absolute_import, print_function, division

import zipfile
import os
import warnings
import glob

import numpy as np
from scipy.io import loadmat
from natsort import natsorted
from pkg_resources import resource_string

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    filter_files,
    run_matlab_cmd,
    download_and_check,
    atomic_directory_setup)
from ..generics import progressinfo

from .utils import create_stimuli, _load


def get_cat2000_test(location=None):
    """
    Loads or downloads and caches the CAT2000 test dataset. The dataset
    consists of 2000 images of
    sizes: 1080 x 1920.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli

    .. seealso::

        Ali Borji, Laurent Itti. CAT2000: A Large Scale Fixation Dataset for Boosting Saliency Research [CVPR 2015 workshop on "Future of Datasets"]

        http://saliency.mit.edu/datasets.html
    """
    if location:
        location = os.path.join(location, 'CAT2000_test')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            return stimuli
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://saliency.mit.edu/testSet.zip',
                               os.path.join(temp_dir, 'testSet.zip'),
                               '903ec668df2e5a8470aef9d8654e7985')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'testSet.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['Output'])
            f.extractall(temp_dir, namelist)

            for filename in ['Pattern/134.jpg']:
                import piexif
                print("Fixing wrong exif rotation tag in '{}'".format(filename))
                full_path = os.path.join(temp_dir, 'testSet', 'Stimuli', filename)
                exif_dict = piexif.load(full_path)
                exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, full_path)

            stimuli_src_location = os.path.join(temp_dir, 'testSet', 'Stimuli')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            images = glob.glob(os.path.join(stimuli_src_location, '**', '*.jpg'))
            images = [i for i in images if 'output' not in i.lower()]
            images = [os.path.relpath(img, start=stimuli_src_location) for img in images]
            stimuli_filenames = natsorted(images)

            stimulus_category_names = [os.path.dirname(filename) for filename in stimuli_filenames]
            category_names = sorted(set(stimulus_category_names))
            categories = [category_names.index(category_name) for category_name in stimulus_category_names]
            attributes = {'category': categories}

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location, attributes=attributes)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
    return stimuli


def get_cat2000_train(location=None, version='1'):
    """
    Loads or downloads and caches the CAT2000 training dataset. The dataset
    consists of 2000 images of
    sizes: 1080 x 1920.
    and the fixations of 18 subjects per image under
    free viewing conditions with 5 seconds presentation time.

    All fixations outside of the image are discarded. This includes
    blinks.

    in v1.1 a few details have been fixed that change the dataset:
    - the quite common repeated fixations are deduplicated (happens serveral thousand times)
    - a values of 1 is substracted from the x and y values to account for the one-based indexing in MATLAB

    To use v1.1, call this function with version='1.1'

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. note::
        This code needs a working matlab or octave installation.

    .. seealso::

        Ali Borji, Laurent Itti. CAT2000: A Large Scale Fixation Dataset for Boosting Saliency Research [CVPR 2015 workshop on "Future of Datasets"]

        http://saliency.mit.edu/datasets.html
    """

    name = 'CAT2000_train'

    if version == '1':
        warnings.warn("You are still using version 1.0 of the CAT2000 dataset implementation in pysaliency. Please upgrade to version 1.1 with pysaliency.get_cat2000_train(..., version='1.1'). This will become the default in pysaliency version 1.0. For more details please see the docstring of this function")
        get_fn = _get_cat2000_train
    elif version == '1.1':
        get_fn = _get_cat2000_train_v1_1
        name += '_v1.1'
    else:
        raise ValueError(version)

    return get_fn(name=name, location=location)


def _get_cat2000_train(name, location):
    """
    Loads or downloads and caches the CAT2000 training dataset. The dataset
    consists of 2000 images of
    sizes: 1080 x 1920.
    and the fixations of 18 subjects per image under
    free viewing conditions with 5 seconds presentation time.

    All fixations outside of the image are discarded. This includes
    blinks.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. note::
        This code needs a working matlab or octave installation.

    .. seealso::

        Ali Borji, Laurent Itti. CAT2000: A Large Scale Fixation Dataset for Boosting Saliency Research [CVPR 2015 workshop on "Future of Datasets"]

        http://saliency.mit.edu/datasets.html
    """
    first_fixation = 0

    if location:
        location = os.path.join(location, name)
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://saliency.mit.edu/trainSet.zip',
                               os.path.join(temp_dir, 'trainSet.zip'),
                               '56ad5c77e6c8f72ed9ef2901628d6e48')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'trainSet.zip'))
            f.extractall(temp_dir)

            stimuli_src_location = os.path.join(temp_dir, 'trainSet', 'Stimuli')
            stimuli_target_location = os.path.join(location, 'Stimuli') if location else None
            images = glob.glob(os.path.join(stimuli_src_location, '**', '*.jpg'))
            images = [os.path.relpath(img, start=stimuli_src_location) for img in images]
            stimuli_filenames = natsorted(images)
            stimulus_category_names = [os.path.dirname(filename) for filename in stimuli_filenames]
            category_names = sorted(set(stimulus_category_names))
            categories = [category_names.index(category_name) for category_name in stimulus_category_names]
            attributes = {'category': categories}

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location, attributes=attributes)

            # FixationTrains

            print('Creating fixations')

            with open(os.path.join(temp_dir, 'load_cat2000.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/{}'.format('load_cat2000.m')))

            # It is vital _not_ to store the extracted fixations in the main
            # directory where matlab is running, as matlab will check the timestamp
            # of all files in this directory very often. This leads to heavy
            # performance penalties and would make matlab run for more than an
            # hour.
            out_path = 'extracted'
            os.makedirs(os.path.join(temp_dir, out_path))
            run_matlab_cmd('load_cat2000;', cwd=temp_dir)

            print('Extracting fixations. This can take some minutes.')
            print('Warning: In the IPython Notebook, the output is shown on the console instead of the notebook.')

            xs = []
            ys = []
            ts = []
            ns = []
            train_subjects = []
            subject_dict = {}

            files = natsorted(glob.glob(os.path.join(temp_dir, out_path, '*.mat')))
            for f in progressinfo(files):
                mat_data = loadmat(f)
                fix_data = mat_data['data']
                name = mat_data['name'][0]
                n = int(os.path.basename(f).split('fix', 1)[1].split('_')[0]) - 1
                stimulus_size = stimuli.sizes[n]
                _, _, subject = name.split('.eye')[0].split('-')
                if subject not in subject_dict:
                    subject_dict[subject] = len(subject_dict)
                subject_id = subject_dict[subject]

                x = []
                y = []
                t = []
                for i in range(first_fixation, fix_data.shape[0]):  # Skip first fixation like Judd does via first_fixation=1
                    if fix_data[i, 0] < 0 or fix_data[i, 1] < 0:
                        continue
                    if fix_data[i, 0] >= stimulus_size[1] or fix_data[i, 1] >= stimulus_size[0]:
                        continue
                    if any(np.isnan(fix_data[i])):  # skip invalid data
                        continue
                    x.append(fix_data[i, 0])
                    y.append(fix_data[i, 1])
                    t.append(len(x))  # Eye Tracker rate = 240Hz
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ns.append(n)
                train_subjects.append(subject_id)
            fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


def _get_cat2000_train_v1_1(name, location):
    """
    Loads or downloads and caches the CAT2000 training dataset. The dataset
    consists of 2000 images of
    sizes: 1080 x 1920.
    and the fixations of 18 subjects per image under
    free viewing conditions with 5 seconds presentation time.

    All fixations outside of the image are discarded. This includes
    blinks.

    in v1_1 a few details have been fixed:
    - the quite common repeated fixations are deduplicated (happens serveral thousand times)
    - a values of 1 is substracted from the x and y values to account for the one-based indexing in MATLAB

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. note::
        This code needs a working matlab or octave installation.

    .. seealso::

        Ali Borji, Laurent Itti. CAT2000: A Large Scale Fixation Dataset for Boosting Saliency Research [CVPR 2015 workshop on "Future of Datasets"]

        http://saliency.mit.edu/datasets.html
    """
    first_fixation = 0

    if location:
        location = os.path.join(location, name)
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://saliency.mit.edu/trainSet.zip',
                               os.path.join(temp_dir, 'trainSet.zip'),
                               '56ad5c77e6c8f72ed9ef2901628d6e48')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'trainSet.zip'))
            f.extractall(temp_dir)

            stimuli_src_location = os.path.join(temp_dir, 'trainSet', 'Stimuli')
            stimuli_target_location = os.path.join(location, 'Stimuli') if location else None
            images = glob.glob(os.path.join(stimuli_src_location, '**', '*.jpg'))
            images = [os.path.relpath(img, start=stimuli_src_location) for img in images]
            stimuli_filenames = natsorted(images)
            stimulus_category_names = [os.path.dirname(filename) for filename in stimuli_filenames]
            category_names = sorted(set(stimulus_category_names))
            categories = [category_names.index(category_name) for category_name in stimulus_category_names]
            attributes = {'category': categories}

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location, attributes=attributes)

            # FixationTrains

            print('Creating fixations')

            with open(os.path.join(temp_dir, 'load_cat2000.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/{}'.format('load_cat2000.m')))

            # It is vital _not_ to store the extracted fixations in the main
            # directory where matlab is running, as matlab will check the timestamp
            # of all files in this directory very often. This leads to heavy
            # performance penalties and would make matlab run for more than an
            # hour.
            out_path = 'extracted'
            os.makedirs(os.path.join(temp_dir, out_path))
            run_matlab_cmd('load_cat2000;', cwd=temp_dir)

            print('Extracting fixations. This can take some minutes.')
            print('Warning: In the IPython Notebook, the output is shown on the console instead of the notebook.')

            xs = []
            ys = []
            ts = []
            ns = []
            train_subjects = []
            subject_dict = {}

            files = natsorted(glob.glob(os.path.join(temp_dir, out_path, '*.mat')))
            for f in progressinfo(files):
                mat_data = loadmat(f)
                fix_data = mat_data['data']
                name = mat_data['name'][0]
                n = int(os.path.basename(f).split('fix', 1)[1].split('_')[0]) - 1
                stimulus_size = stimuli.sizes[n]
                _, _, subject = name.split('.eye')[0].split('-')
                if subject not in subject_dict:
                    subject_dict[subject] = len(subject_dict)
                subject_id = subject_dict[subject]

                x = []
                y = []
                t = []
                for i in range(first_fixation, fix_data.shape[0]):
                    new_x = fix_data[i, 0] - 1  # one-based indexing in MATLAB
                    new_y = fix_data[i, 1] - 1  # one-based indexing in MATLAB
                    if new_x < 0 or new_y < 0:
                        continue
                    if new_x >= stimulus_size[1] or new_y >= stimulus_size[0]:
                        continue
                    if any(np.isnan(fix_data[i])):  # skip invalid data
                        continue

                    if x and x[-1] == new_x and y[-1] == new_y:
                        # repeated fixation, skip
                        continue

                    x.append(new_x)
                    y.append(new_y)
                    t.append(len(x))  # Eye Tracker rate = 240Hz
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ns.append(n)
                train_subjects.append(subject_id)
            fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects)

        print(subject_dict)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations