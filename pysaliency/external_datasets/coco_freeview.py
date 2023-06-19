import glob
from hashlib import md5
import json
import os
import shutil
from subprocess import check_call
import zipfile


import numpy as np
from PIL import Image
from tqdm import tqdm

from ..datasets import FixationTrains, create_subset
from ..utils import (
    TemporaryDirectory,
    filter_files,
    download_and_check,
    atomic_directory_setup)

from .utils import create_stimuli, _load
from .coco_search18 import _prepare_stimuli


def get_COCO_Freeview(location=None):
    """
    Loads or downloads and caches the COCO Freeview dataset.

    The dataset consists of about 5317 images from MS COCO with
    scanpath data from 10 observers doing freeviewing.

    The COCO images have been rescaled and padded to a size of
    1680x1050 pixels.

    The scanpaths come with attributes for
    - (fixation) duration in seconds

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `COCO-Search18` of
                     location and read from there, if already present.
    @return: Training stimuli, training FixationTrains, validation Stimuli, validation FixationTrains

    .. seealso::

        Chen, Y., Yang, Z., Chakraborty, S., Mondal, S., Ahn, S., Samaras, D., Hoai, M., & Zelinsky, G. (2022).
        Characterizing Target-Absent Human Attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (pp. 5031-5040).

        Yang, Z., Mondal, S., Ahn, S., Zelinsky, G., Hoai, M., & Samaras, D. (2023).
        Predicting Human Attention using Computational Attention. arXiv preprint arXiv:2303.09383.
    """

    if location:
        location = os.path.join(location, 'COCO-Freeview')
        if os.path.exists(location):
            stimuli_train = _load(os.path.join(location, 'stimuli_train.hdf5'))
            fixations_train = _load(os.path.join(location, 'fixations_train.hdf5'))
            stimuli_validation = _load(os.path.join(location, 'stimuli_validation.hdf5'))
            fixations_validation = _load(os.path.join(location, 'fixations_validation.hdf5'))
            return stimuli_train, fixations_train, stimuli_validation, fixations_validation
        os.makedirs(location)

    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip',
                               os.path.join(temp_dir, 'COCOSearch18-images-TP.zip'),
                               '4a815bb591cb463ab77e5ba0c68fedfb')

            download_and_check('http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TA.zip',
                               os.path.join(temp_dir, 'COCOSearch18-images-TA.zip'),
                               '85af7d74fa57c202320fa5e7d0dcc187')

            download_and_check('http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json',
                               os.path.join(temp_dir, 'COCOFreeView_fixations_trainval.json'),
                               'd43d3e22de7b73297b3b35cb24d12c79')


            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'COCOSearch18-images-TP.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            f = zipfile.ZipFile(os.path.join(temp_dir, 'COCOSearch18-images-TA.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            # unifying images for different tasks

            stimulus_directory = os.path.join(temp_dir, 'stimuli')
            os.makedirs(stimulus_directory)

            filenames, stimulus_tasks = _prepare_stimuli(temp_dir, stimulus_directory, merge_tasks=True, unique_images=False)

            stimuli_src_location = os.path.join(temp_dir, 'stimuli')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            stimuli_filenames = filenames
            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            print('creating fixations')

            with open(os.path.join(temp_dir, 'COCOFreeView_fixations_trainval.json')) as fixation_file:
                json_data = json.load(fixation_file)

            all_scanpaths = _get_COCO_Freeview_fixations(json_data, filenames)

            scanpaths_train = all_scanpaths.filter_fixation_trains(all_scanpaths.scanpath_attributes['split'] == 'train')
            scanpaths_validation = all_scanpaths.filter_fixation_trains(all_scanpaths.scanpath_attributes['split'] == 'valid')

            del scanpaths_train.scanpath_attributes['split']
            del scanpaths_validation.scanpath_attributes['split']

            ns_train = sorted(set(scanpaths_train.n))
            stimuli_train, fixations_train = create_subset(stimuli, scanpaths_train, ns_train)

            ns_val = sorted(set(scanpaths_validation.n))
            stimuli_val, fixations_val = create_subset(stimuli, scanpaths_validation, ns_val)

        if location:
            stimuli_train.to_hdf5(os.path.join(location, 'stimuli_train.hdf5'))
            fixations_train.to_hdf5(os.path.join(location, 'fixations_train.hdf5'))
            stimuli_val.to_hdf5(os.path.join(location, 'stimuli_validation.hdf5'))
            fixations_val.to_hdf5(os.path.join(location, 'fixations_validation.hdf5'))

    return stimuli_train, fixations_train, stimuli_val, fixations_val


def get_COCO_Freeview_train(location=None):
    stimuli_train, fixations_train, stimuli_val, fixations_val = get_COCO_Freeview(location=location)
    return stimuli_train, fixations_train


def get_COCO_Freeview_validation(location=None):
    stimuli_train, fixations_train, stimuli_val, fixations_val = get_COCO_Freeview(location=location)
    return stimuli_val, fixations_val


def _get_COCO_Freeview_fixations(json_data, filenames):
    train_xs = []
    train_ys = []
    train_ts = []
    train_ns = []
    train_subjects = []
    train_durations = []
    split = []

    for item in tqdm(json_data):
        filename = item['name']
        n = filenames.index(filename)

        train_xs.append(item['X'])
        train_ys.append(item['Y'])
        train_ts.append(np.arange(item['length']))
        train_ns.append(n)
        train_subjects.append(item['subject'])
        train_durations.append(np.array(item['T']) / 1000)
        split.append(item['split'])

    scanpath_attributes = {
        'split': split,
    }
    scanpath_fixation_attributes = {
        'durations': train_durations,
    }
    scanpath_attribute_mapping = {
        'durations': 'duration'
    }
    fixations = FixationTrains.from_fixation_trains(
        train_xs,
        train_ys,
        train_ts,
        train_ns,
        train_subjects,
        scanpath_attributes=scanpath_attributes,
        scanpath_fixation_attributes=scanpath_fixation_attributes,
        scanpath_attribute_mapping=scanpath_attribute_mapping,
    )

    return fixations