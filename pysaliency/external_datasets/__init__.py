from __future__ import absolute_import, print_function, division

import zipfile
import os
import shutil
import warnings
import glob
import itertools
from six.moves import urllib

import numpy as np
from scipy.io import loadmat
from natsort import natsorted
import dill
from pkg_resources import resource_string
from PIL import Image
from boltons.fileutils import mkdir_p
from tqdm import tqdm

from ..datasets import FileStimuli, Stimuli, FixationTrains, Fixations, read_hdf5
from ..utils import (
    TemporaryDirectory,
    filter_files,
    run_matlab_cmd,
    download_and_check,
    download_file_from_google_drive,
    check_file_hash,
    atomic_directory_setup,
    build_padded_2d_array)
from ..generics import progressinfo

from .utils import create_memory_stimuli, create_stimuli, _load

from .toronto import get_toronto, get_toronto_with_subjects
from .mit import get_mit1003, get_mit1003_with_initial_fixation, get_mit1003_onesize, get_mit300
from .cat2000 import get_cat2000_test, get_cat2000_train
from .isun import get_iSUN, get_iSUN_training, get_iSUN_validation, get_iSUN_testing


def _get_SALICON_obsolete(data_type, stimuli_url, stimuli_hash, fixation_url, fixation_hash, location):
    """Obsolete code for loading from the SALICON json files. The URLs don't exist anymore and the LSUN challenge uses simpler mat files"""
    from salicon.salicon import SALICON
    with TemporaryDirectory(cleanup=True) as temp_dir:
        download_and_check(stimuli_url,
                           os.path.join(temp_dir, 'stimuli.zip'),
                           stimuli_hash)
        if fixation_url is not None:
            download_and_check(fixation_url,
                               os.path.join(temp_dir, 'fixations.json'),
                               fixation_hash)
        # Stimuli

        annFile = os.path.join(temp_dir, 'fixations.json')
        salicon = SALICON(annFile)

        # get all images
        imgIds = salicon.getImgIds()
        images = salicon.loadImgs(imgIds)

        print('Creating stimuli')
        f = zipfile.ZipFile(os.path.join(temp_dir, 'stimuli.zip'))
        namelist = f.namelist()
        f.extractall(temp_dir, namelist)
        del f

        stimuli_src_location = os.path.join(temp_dir, data_type)
        stimuli_target_location = os.path.join(location, 'stimuli') if location else None
        filenames = [img['file_name'] for img in images]
        stimuli = create_stimuli(stimuli_src_location, filenames, stimuli_target_location)

        if fixation_url is not None:
            # FixationTrains
            print('Creating fixations')

            ns = []
            train_xs = []
            train_ys = []
            train_ts = []
            train_subjects = []

            for n, imgId in progressinfo(enumerate(imgIds)):
                annIds = salicon.getAnnIds(imgIds=imgId)
                anns = salicon.loadAnns(annIds)
                for ann in anns:
                    fs = ann['fixations']
                    # SALICON annotations are 1-indexed, not 0-indexed.
                    xs = np.array([f[1] - 1 for f in fs])
                    ys = np.array([f[0] - 1 for f in fs])
                    ns.append(np.ones(len(xs), dtype=int) * n)
                    train_xs.append(xs)
                    train_ys.append(ys)
                    train_ts.append(range(len(xs)))
                    train_subjects.append(np.ones(len(xs), dtype=int) * ann['worker_id'])

            xs = np.hstack(train_xs)
            ys = np.hstack(train_ys)
            ts = np.hstack(train_ts)

            ns = np.hstack(ns)
            subjects = np.hstack(train_subjects)

            fixations = Fixations.FixationsWithoutHistory(xs, ys, ts, ns, subjects)
        else:
            fixations = None

    if location:
        stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
        if fixations is not None:
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


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
    stimuli_train, stimuli_val, stimuli_test, fixations_train, fixations_val = get_SALICON(location=location, edition=edition, fixation_type=fixation_type)

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
    stimuli_train, stimuli_val, stimuli_test, fixations_train, fixations_val = get_SALICON(location=location, edition=edition, fixation_type=fixation_type)

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
    stimuli_train, stimuli_val, stimuli_test, fixations_train, fixations_val = get_SALICON(location=location, edition=edition, fixation_type=fixation_type)

    return stimuli_test


def _get_koehler_fixations(data, task, n_stimuli):
    tasks = {'freeviewing': 'freeview',
             'objectsearch': 'objsearch',
             'saliencysearch': 'salview'}
    task = tasks[task]

    data_x = data['{}_x'.format(task)]
    data_y = data['{}_y'.format(task)]

    # Load Fixation Data
    xs = []
    ys = []
    ts = []
    ns = []
    subjects = []
    subject_ids = range(data_x.shape[0])

    for n, subject_id in tqdm(list(itertools.product(range(n_stimuli), subject_ids))):
        x = data_x[subject_id, n, :] - 1
        y = data_y[subject_id, n, :] - 1
        inds = np.logical_not(np.isnan(x))
        x = x[inds]
        y = y[inds]
        xs.append(x)
        ys.append(y)
        ts.append(range(len(x)))
        ns.append(n)
        subjects.append(subject_id)
    return FixationTrains.from_fixation_trains(xs, ys, ts, ns, subjects)


def get_koehler(location=None, datafile=None):
    """
    Loads or or extracts and caches the Koehler dataset. The dataset
    consists of 800 color images of outdoor and indoor scenes
    of size 405x405px and the fixations for three different tasks:
    free viewing, object search and saliency search.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `koehler` of
                     location and read from there, if already present.
    @return: stimuli, fixations_freeviewing, fixations_objectsearch, fixations_saliencysearch

    .. note:: As this dataset is only after filling a download form, pysaliency
              cannot download it for you. Instead you have to download the file
              `PublicData.zip` and provide it to this function via the `datafile`
              keyword argument on the first call.

    .. seealso::

        Kathryn Koehler, Fei Guo, Sheng Zhang, Miguel P. Eckstein. What Do Saliency Models Predict? [JoV 2014]

        http://www.journalofvision.org/content/14/3/14.full

        https://labs.psych.ucsb.edu/eckstein/miguel/research_pages/saliencydata.html
    """
    if location:
        location = os.path.join(location, 'Koehler')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations_freeviewing = _load(os.path.join(location, 'fixations_freeviewing.hdf5'))
            fixations_objectsearch = _load(os.path.join(location, 'fixations_objectsearch.hdf5'))
            fixations_saliencysearch = _load(os.path.join(location, 'fixations_saliencysearch.hdf5'))
            return stimuli, fixations_freeviewing, fixations_objectsearch, fixations_saliencysearch
        mkdir_p(location)
    if not datafile:
        raise ValueError('The Koehler dataset is not freely available! You have to '
                         'request the data file from the authors and provide it to '
                         'this function via the datafile argument')
    with atomic_directory_setup(location):
        with TemporaryDirectory() as temp_dir:
            z = zipfile.ZipFile(datafile)
            print('Extracting')
            z.extractall(temp_dir)

            # Stimuli
            stimuli_src_location = os.path.join(temp_dir, 'Images')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            stimuli_filenames = ['image_r_{}.jpg'.format(i) for i in range(1, 801)]

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            # Fixations

            data = loadmat(os.path.join(temp_dir, 'ObserverData.mat'))

            fs = []

            for task in ['freeviewing', 'objectsearch', 'saliencysearch']:
                fs.append(_get_koehler_fixations(data, task, len(stimuli)))

            if location:
                stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
                fs[0].to_hdf5(os.path.join(location, 'fixations_freeviewing.hdf5'))
                fs[1].to_hdf5(os.path.join(location, 'fixations_objectsearch.hdf5'))
                fs[2].to_hdf5(os.path.join(location, 'fixations_saliencysearch.hdf5'))
    return [stimuli] + fs


def _load_FIGRIM_data(filename, stimuli_indices, stimulus_type):
    data = loadmat(filename)['allImages'].flatten()
    xs = []
    ys = []
    ts = []
    ns = []
    train_subjects = []
    which_times = []
    which_time_names = ['enc', 'rec', 'rec2']
    stimulus_types = []
    responses = []

    for stimulus_data in data:
        n = stimuli_indices[stimulus_data['filename'][0]]
        # category = stimulus_data['category'][0]  # TODO: use
        for subject, subject_data in enumerate(stimulus_data['userdata'].flatten()):
            if not subject_data['trial']:
                # No data for this subject and this stimulus
                continue

            for which_time in which_time_names:
                fixations = subject_data['fixations'][0, 0][which_time]
                if not len(fixations):
                    continue
                # if len(fixations) and which_time != 'enc':
                #     print("Problem:", n, subject_name, which_time)
                subject_response = subject_data['SDT'][0][which_time_names.index(which_time)]

                xs.append(fixations[:, 0])
                ys.append(fixations[:, 1])
                ts.append(np.arange(len(xs[-1])))
                ns.append(n)
                train_subjects.append(subject)

                which_times.append(which_time_names.index(which_time))
                stimulus_types.append(stimulus_type)
                responses.append(subject_response)

    return xs, ys, ts, ns, train_subjects, which_times, stimulus_types, responses


def get_FIGRIM(location=None):
    """
    Loads or downloads and caches the FIGRIM dataset. The dataset
    consists of >2700 scenes of sizes 1000x1000px
    and the fixations of subjects while doing a repetition
    recognition task with 3 seconds presentation time.
    subject responses etc are included.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. note::
        This dataset comes with additional annotations:
            - stimulus_type: 0=filler, 1=target
            - which_time: 0=encoding, 1=first recognition, 2=second recognition
            - response: 1=hit, 2=false alarm, 3=miss, 4=correct rejection

    .. seealso::

        Bylinskii, Zoya and Isola, Phillip and Bainbridge, Constance and Torralba, Antonio and Oliva, Aude. Intrinsic and Extrinsic Effects on Image Memorability [Vision research 2015]

        http://figrim.mit.edu/index_eyetracking.html
    """
    if location:
        location = os.path.join(location, 'FIGRIM')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://figrim.mit.edu/Fillers.zip',
                               os.path.join(temp_dir, 'Fillers.zip'),
                               'dc0bc9561b5bc90e158ec32074dd1060')

            download_and_check('http://figrim.mit.edu/Targets.zip',
                               os.path.join(temp_dir, 'Targets.zip'),
                               '2ad3a42ebc377efe4b39064405568201')

            download_and_check('https://github.com/cvzoya/figrim/blob/master/targetData/allImages_release.mat?raw=True',
                               os.path.join(temp_dir, 'allImages_release.mat'),
                               'c72843b05e95ab27594c1d11c849c897')

            download_and_check('https://github.com/cvzoya/figrim/blob/master/fillerData/allImages_fillers.mat?raw=True',
                               os.path.join(temp_dir, 'allImages_fillers.mat'),
                               'ce4f8b4961005d62f7a21191a64cab5e')

            # Stimuli
            mkdir_p(os.path.join(temp_dir, 'stimuli'))
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'Fillers.zip'))
            f.extractall(os.path.join(temp_dir, 'stimuli'))

            f = zipfile.ZipFile(os.path.join(temp_dir, 'Targets.zip'))
            f.extractall(os.path.join(temp_dir, 'stimuli'))

            stimuli_src_location = os.path.join(temp_dir, 'stimuli')
            stimuli_target_location = os.path.join(location, 'Stimuli') if location else None
            images = glob.glob(os.path.join(stimuli_src_location, '**', '**', '*.jpg'))
            images = [os.path.relpath(img, start=stimuli_src_location) for img in images]
            stimuli_filenames = natsorted(images)

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            stimuli_basenames = [os.path.basename(filename) for filename in stimuli_filenames]
            stimulus_indices = {s: stimuli_basenames.index(s) for s in stimuli_basenames}

            # FixationTrains

            print('Creating fixations')

            print('Fillers...')
            (xs_filler,
             ys_filler,
             ts_filler,
             ns_filler,
             train_subjects_filler,
             which_times_filler,
             stimulus_types_filler,
             responses_filler) = _load_FIGRIM_data(os.path.join(temp_dir, 'allImages_fillers.mat'), stimulus_indices, stimulus_type=0)

            print("Targets...")
            (xs_target,
             ys_target,
             ts_target,
             ns_target,
             train_subjects_target,
             which_times_target,
             stimulus_types_target,
             responses_target) = _load_FIGRIM_data(os.path.join(temp_dir, 'allImages_release.mat'), stimulus_indices, stimulus_type=0)

            print("Finalizing...")
            xs = xs_filler + xs_target
            ys = ys_filler + ys_target
            ts = ts_filler + ts_target
            ns = ns_filler + ns_target
            train_subjects = train_subjects_filler + train_subjects_target
            which_times = which_times_filler + which_times_target
            stimulus_types = stimulus_types_filler + stimulus_types_target
            responses = responses_filler + responses_target

            fixations = FixationTrains.from_fixation_trains(
                xs, ys, ts, ns, train_subjects,
                attributes={
                    'which_time': which_times,
                    'stimulus_type': stimulus_types,
                    'response': responses
                })

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


def get_OSIE(location=None):
    """
    Loads or downloads and caches the OSIE dataset. The dataset
    consists of 700 images of size 800x600px
    and the fixations of 15 subjects while doing a
    freeviewing task with 3 seconds presentation time.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. seealso::

        Juan Xu, Ming Jiang, Shuo Wang, Mohan Kankanhalli, Qi Zhao. Predicting Human Gaze Beyond Pixels [JoV 2014]

        http://www-users.cs.umn.edu/~qzhao/predicting.html
    """
    if location:
        location = os.path.join(location, 'OSIE')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            stimuli_src_location = os.path.join(temp_dir, 'stimuli')
            mkdir_p(stimuli_src_location)
            images = []
            for i in tqdm(list(range(700))):
                filename = '{}.jpg'.format(i + 1001)
                target_name = os.path.join(stimuli_src_location, filename)
                urllib.request.urlretrieve(
                    'https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/raw/master/data/stimuli/' + filename,
                    target_name)
                images.append(filename)

            download_and_check('https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/blob/master/data/eye/fixations.mat?raw=true',
                               os.path.join(temp_dir, 'fixations.mat'),
                               '8efdf6fe66f38b6e70f854c7ff45aa70')

            # Stimuli
            print('Creating stimuli')

            stimuli_target_location = os.path.join(location, 'Stimuli') if location else None
            stimuli = create_stimuli(stimuli_src_location, images, stimuli_target_location)

            stimulus_indices = {s: images.index(s) for s in images}

            # FixationTrains

            print('Creating fixations')
            data = loadmat(os.path.join(temp_dir, 'fixations.mat'))['fixations'].flatten()

            xs = []
            ys = []
            ts = []
            ns = []
            train_subjects = []

            for stimulus_data in data:
                stimulus_data = stimulus_data[0, 0]
                n = stimulus_indices[stimulus_data['img'][0]]
                for subject, subject_data in enumerate(stimulus_data['subjects'].flatten()):
                    fixations = subject_data[0, 0]
                    if not len(fixations['fix_x'].flatten()):
                        continue

                    xs.append(fixations['fix_x'].flatten())
                    ys.append(fixations['fix_y'].flatten())
                    ts.append(np.arange(len(xs[-1])))
                    ns.append(n)
                    train_subjects.append(subject)

        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


def get_NUSEF_public(location=None):
    """
    Loads or downloads and caches the part of the NUSEF dataset,
    for which the stimuli are public. The dataset
    consists of 758 images of size 1024x700px
    and the fixations of 25 subjects while doing a
    freeviewing task with 5 seconds presentation time.

    Part of the stimuli from NUSEF are available only
    under a special license and only upon request. This
    function returns only the 444 images which are
    available public (and the corresponding fixations).

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. seealso::

        Subramanian Ramanathan, Harish Katti, Nicu Sebe, Mohan Kankanhalli, Tat-Seng Chua. An eye fixation database for saliency detection in images [ECCV 2010]

        http://mmas.comp.nus.edu.sg/NUSEF.html
    """
    if location:
        location = os.path.join(location, 'NUSEF_public')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:

            download_and_check('http://mmas.comp.nus.edu.sg/NUSEF_database.zip',
                               os.path.join(temp_dir, 'NUSEF_database.zip'),
                               '429a78ad92184e8a4b37419988d98953')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'NUSEF_database.zip'))
            f.extractall(temp_dir)

            stimuli_src_location = os.path.join(temp_dir, 'NUSEF_database', 'stimuli')
            images = glob.glob(os.path.join(stimuli_src_location, '*.jpg'))
            images = [os.path.relpath(img, start=stimuli_src_location) for img in images]
            stimuli_filenames = sorted(images)

            stimuli_target_location = os.path.join(location, 'Stimuli') if location else None
            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            stimuli_basenames = [os.path.basename(f) for f in stimuli_filenames]
            stimuli_indices = {s: stimuli_basenames.index(s) for s in stimuli_basenames}

            # FixationTrains

            print('Creating fixations')

            xs = []
            ys = []
            ts = []
            ns = []
            train_subjects = []

            scale_x = 1024 / 260
            scale_y = 768 / 280

            fix_location = os.path.join(temp_dir, 'NUSEF_database', 'fix_data')
            for sub_dir in tqdm(os.listdir(fix_location)):
                if not sub_dir + '.jpg' in stimuli_indices:
                    # one of the non public images
                    continue
                n = stimuli_indices[sub_dir + '.jpg']
                for subject_data in glob.glob(os.path.join(fix_location, sub_dir, '*.fix')):
                    data = open(subject_data).read().replace('\r\n', '\n')
                    data = data.split('COLS=', 1)[1]
                    data = data.split('[Fix Segment Summary')[0]
                    lines = data.split('\n')[1:]
                    lines = [l for l in lines if l]
                    x = []
                    y = []
                    t = []
                    for line in lines:
                        (_,
                         seg_no,
                         fix_no,
                         pln_no,
                         start_time,
                         fix_dur,
                         interfix_dur,
                         interfix_deg,
                         hor_pos,
                         ver_pos,
                         pupil_diam,
                         eye_scn_dist,
                         no_of_flags,
                         fix_loss,
                         interfix_loss) = line.split()
                        x.append(float(hor_pos) * scale_x)
                        y.append(float(ver_pos) * scale_y)
                        t.append(float(start_time.split(':')[-1]))

                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                    ns.append(n)
                    train_subjects.append(0)

        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


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
