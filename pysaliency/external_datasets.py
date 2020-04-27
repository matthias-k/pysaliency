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

from .datasets import FileStimuli, Stimuli, FixationTrains, Fixations, read_hdf5
from .utils import (
    TemporaryDirectory,
    filter_files,
    run_matlab_cmd,
    download_and_check,
    download_file_from_google_drive,
    check_file_hash,
    atomic_directory_setup,
    build_padded_2d_array)
from .generics import progressinfo


def create_memory_stimuli(filenames, attributes=None):
    """
    Create a `Stimuli`-class from a list of filenames by reading the them
    """
    tmp_stimuli = FileStimuli(filenames)
    stimuli = list(tmp_stimuli.stimuli)  # Read all stimuli
    return Stimuli(stimuli, attributes=attributes)


def create_stimuli(stimuli_location, filenames, location=None, attributes=None):
    """
    Create a Stimuli class of stimuli.

    Parameters
    ----------

    @type  stimuli_location: string
    @param stimuli_location: the base path where the stimuli are located.
                             If `location` is provided, this directory will
                             be copied to `location` (see below).

    @type  filenames: list of strings
    @param filenames: lists the filenames of the stimuli to include in the dataset.
                      Filenames have to be relative to `stimuli_location`.

    @type  location: string or `None`
    @param location: If provided, the function will copy the filenames to
                     `location` and return a `FileStimuli`-object for the
                     copied files. Otherwise a `Stimuli`-object is returned.

    @returns: `Stimuli` or `FileStimuli` object depending on `location`.

    """
    if location is not None:
        shutil.copytree(stimuli_location,
                        location)
        filenames = [os.path.join(location, f) for f in filenames]

        return FileStimuli(filenames, attributes=attributes)

    else:
        filenames = [os.path.join(stimuli_location, f) for f in filenames]
        return create_memory_stimuli(filenames, attributes=attributes)


def _load(filename):
    """attempt to load hdf5 file and fallback to pickle files if present"""
    if os.path.isfile(filename):
        return read_hdf5(filename)

    stem, ext = os.path.splitext(filename)
    pydat_filename = stem + '.pydat'

    return dill.load(open(pydat_filename, 'rb'))


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
    @return: Stimuli, FixationTrains

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
            ts = []
            xs = []
            ys = []
            ns = []
            subjects = []
            for n in range(len(stimuli.stimuli)):
                _ys, _xs = np.nonzero(points[0, n])
                xs.extend([[x] for x in _xs])
                ys.extend([[y] for y in _ys])
                ns.extend([n for x in _xs])
                ts.extend([[0] for x in _xs])
                subjects.extend([0 for x in _xs])
            fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, subjects)
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
                    content = open(os.path.join(raw_path, subject, imagename + '.fix')).read()
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


def _get_mit1003(dataset_name, location=None, include_initial_fixation=False, only_1024_by_768=False):
    """
    .. seealso::

        Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict where Humans Look [ICCV 2009]

        http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
    """

    if include_initial_fixation:
        first_fixation = 0
    else:
        first_fixation = 1

    if location:
        location = os.path.join(location, dataset_name)
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip',
                               os.path.join(temp_dir, 'ALLSTIMULI.zip'),
                               '0d7df8b954ecba69b6796e77b9afe4b6')
            download_and_check('http://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip',
                               os.path.join(temp_dir, 'DATA.zip'),
                               'ea19d74ad0a0144428c53e9d75c2d71c')
            download_and_check('http://people.csail.mit.edu/tjudd/WherePeopleLook/Code/DatabaseCode.zip',
                               os.path.join(temp_dir, 'DatabaseCode.zip'),
                               'd8e5e2b6ec827f4115ddbff59b0bdf1d')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'ALLSTIMULI.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            stimuli_src_location = os.path.join(temp_dir, 'ALLSTIMULI')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            images = glob.glob(os.path.join(stimuli_src_location, '*.jpeg'))
            images = [os.path.split(img)[1] for img in images]
            stimuli_filenames = natsorted(images)

            if only_1024_by_768:
                def check_size(f):
                    img = Image.open(os.path.join(stimuli_src_location, f))
                    return img.size == (1024, 768)

                stimuli_filenames = [s for s in stimuli_filenames if check_size(s)]

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            # FixationTrains

            print('Creating fixations')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'DATA.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            f = zipfile.ZipFile(os.path.join(temp_dir, 'DatabaseCode.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            subjects = glob.glob(os.path.join(temp_dir, 'DATA', '*'))
            # Exclude files
            subjects = [s for s in subjects if not os.path.splitext(s)[1]]
            subjects = [os.path.basename(s) for s in subjects]
            subjects = sorted(subjects)

            with open(os.path.join(temp_dir, 'extract_fixations.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/{}'.format('extract_fixations.m')))

            cmds = []
            # It is vital _not_ to store the extracted fixations in the main
            # directory where matlab is running, as matlab will check the timestamp
            # of all files in this directory very often. This leads to heavy
            # performance penalties and would make matlab run for more than an
            # hour.
            out_path = 'extracted'
            os.makedirs(os.path.join(temp_dir, out_path))
            total_cmd_count = len(stimuli_filenames) * len(subjects)
            for n, stimulus in enumerate(stimuli_filenames):
                for subject_id, subject in enumerate(subjects):
                    subject_path = os.path.join('DATA', subject)
                    outfile = '{0}_{1}.mat'.format(stimulus, subject)
                    outfile = os.path.join(out_path, outfile)
                    cmds.append("fprintf('%d/%d\\n', {}, {});".format(n * len(subjects) + subject_id, total_cmd_count))
                    cmds.append("extract_fixations('{0}', '{1}', '{2}');".format(stimulus, subject_path, outfile))

            print('Running original code to extract fixations. This can take some minutes.')
            print('Warning: In the IPython Notebook, the output is shown on the console instead of the notebook.')
            with open(os.path.join(temp_dir, 'extract_all_fixations.m'), 'w') as f:
                for cmd in cmds:
                    f.write('{}\n'.format(cmd))

            run_matlab_cmd('extract_all_fixations;', cwd=temp_dir)
            xs = []
            ys = []
            ts = []
            ns = []
            train_subjects = []
            duration_hist = []
            train_durations = []
            for n, stimulus in enumerate(stimuli_filenames):
                stimulus_size = stimuli.sizes[n]
                for subject_id, subject in enumerate(subjects):
                    subject_name = os.path.split(subject)[-1]
                    outfile = '{0}_{1}.mat'.format(stimulus, subject_name)
                    mat_data = loadmat(os.path.join(temp_dir, out_path, outfile))
                    fix_data = mat_data['fixations']
                    starts = mat_data['starts']
                    _durations = mat_data['durations']
                    x = []
                    y = []
                    t = []
                    duration = []
                    # if first_fixation == 1 the first fixation is skipped, as done
                    # by Judd.
                    # TODO: This contains a subtle bug: if the first fixation is invalid,
                    # the next fixation is not fixed. Therefore, the dataset still
                    # contains a few initial fixations.
                    for i in range(first_fixation, fix_data.shape[0]):
                        if fix_data[i, 0] < 0 or fix_data[i, 1] < 0:
                            continue
                        if fix_data[i, 0] >= stimulus_size[1] or fix_data[i, 1] >= stimulus_size[0]:
                            continue
                        x.append(fix_data[i, 0])
                        y.append(fix_data[i, 1])
                        t.append(starts[0, i] / 240.0)  # Eye Tracker rate = 240Hz
                        duration_hist.append(np.array(duration))
                        duration.append(_durations[0, i] / 1000)  # data is in ms, we want seconds
                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                    ns.append(n)
                    train_subjects.append(subject_id)
                    train_durations.append(duration)

            attributes = {
                'duration_hist': build_padded_2d_array(duration_hist),
            }
            scanpath_attributes = {
                'train_durations': build_padded_2d_array(train_durations),
            }
            fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects, attributes=attributes, scanpath_attributes=scanpath_attributes)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


def get_mit1003(location=None):
    """
    Loads or downloads and caches the MIT1003 dataset. The dataset
    consists of 1003 natural indoor and outdoor scenes of
    sizes: max dim: 1024px, other dim: 405-1024px
    and the fixations of 15 subjects under
    free viewing conditions with 3 seconds presentation time.

    All fixations outside of the image are discarded. This includes
    blinks.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. note::
        This code needs a working matlab or octave installation as the original
        matlab code by Judd et al. is used to extract the fixation from the
        eyetracking data.

        The first fixation of each fixation train is discarded as stated in the
        paper (Judd et al. 2009).

    .. seealso::

        Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict where Humans Look [ICCV 2009]

        http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
    """
    return _get_mit1003('MIT1003', location=location, include_initial_fixation=False)


def get_mit1003_with_initial_fixation(location=None):
    """
    Loads or downloads and caches the MIT1003 dataset. The dataset
    consists of 1003 natural indoor and outdoor scenes of
    sizes: max dim: 1024px, other dim: 405-1024px
    and the fixations of 15 subjects under
    free viewing conditions with 3 seconds presentation time.

    All fixations outside of the image are discarded. This includes
    blinks.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. note::
        This code needs a working matlab or octave installation as the original
        matlab code by Judd et al. is used to extract the fixation from the
        eyetracking data.

        Unlike in the original paper (Judd et al. 2009), in this version of
        the MIT1003 dataset the first fixation of each scanpath
        is *not* discarded. See `get_mit1003` if you want to use
        the dataset as suggested by Judd.

    .. seealso::

        Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict where Humans Look [ICCV 2009]

        http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
    """
    return _get_mit1003('MIT1003_initial_fix', location=location, include_initial_fixation=True)


def get_mit1003_onesize(location=None):
    """
    Loads or downloads and caches the subset of the MIT1003 dataset
    used in "How close are we to understanding image-based
    saliency" (http://arxiv.org/abs/1409.7686) and
    "Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained
    on ImageNet" (http://arxiv.org/abs/1411.1045).
    The dataset conists of 463 natural indoor and outdoor scenes of
    size 1024 x 768 px and the fixations of 15 subjects under
    free viewing conditions with 3 seconds presentation time.

    All fixations outside of the image are discarded. This includes
    blinks.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. note::
        This code needs a working matlab or octave installation as the original
        matlab code by Judd et al. is used to extract the fixation from the
        eyetracking data.

        The first fixation of each fixation train is discarded as stated in the
        paper (Judd et al. 2009).

    .. seealso::

        Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict where Humans Look [ICCV 2009]

        http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
    """
    return _get_mit1003('MIT1003_onesize', location=location, include_initial_fixation=False, only_1024_by_768=True)


def get_mit300(location=None):
    """
    Loads or downloads and caches the MIT300 test stimuli for
    the MIT saliency benchmark. The dataset
    consists of 300 stimuli
    and the fixations of 40 subjects under
    free viewing conditions with 3 seconds presentation time.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli

    .. seealso::

        Zoya Bylinskii and Tilke Judd and Ali Borji and Laurent Itti and Fr{\'e}do Durand and Aude Oliva and Antonio Torralba. MIT Saliency Benchmark

        http://saliency.mit.edu
    """
    if location:
        location = os.path.join(location, 'MIT300')
        if os.path.exists(location):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            return stimuli
        os.makedirs(location)
    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://saliency.mit.edu/BenchmarkIMAGES.zip',
                               os.path.join(temp_dir, 'BenchmarkIMAGES.zip'),
                               '03ed32bdf5e4289950cd28df89451260')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'BenchmarkIMAGES.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            stimuli_src_location = os.path.join(temp_dir, 'BenchmarkIMAGES')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            images = glob.glob(os.path.join(stimuli_src_location, '*.jpg'))
            images = [os.path.split(img)[1] for img in images]
            stimuli_filenames = natsorted(images)

            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
    return stimuli


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


def get_cat2000_train(location=None, include_initial_fixation=True):
    name = 'CAT2000_train'

    if not include_initial_fixation:
        name += '_without_initial_fixation'

    return _get_cat2000_train(name=name, location=location, include_initial_fixation=include_initial_fixation)


def _get_cat2000_train(name, location, include_initial_fixation):
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
    if include_initial_fixation:
        first_fixation = 0
    else:
        first_fixation = 1

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
    training_stimuli, validation_stimuli, test_stimuli, training_fixations, validation_fixations = get_iSUN(location=location)
    return training_stimuli, training_fixations


def get_iSUN_validation(location=None):
    """
    @return: validation stimuli, validation fixation trains

    See `get_iSUN` for more information"""
    training_stimuli, validation_stimuli, test_stimuli, training_fixations, validation_fixations = get_iSUN(location=location)
    return validation_stimuli, validation_fixations


def get_iSUN_testing(location=None):
    """
    @return: testing stimuli

    See `get_iSUN` for more information"""
    training_stimuli, validation_stimuli, test_stimuli, training_fixations, validation_fixations = get_iSUN(location=location)
    return test_stimuli


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
                download_file_from_google_drive('0B2hsWbciDVedWHFiMUVVWFRZTE0', fixations_file)
                check_file_hash(fixations_file, '9a22db9d718200fb90252e5010c004c4')
            elif edition == '2017':
                download_file_from_google_drive('0B2hsWbciDVedS1lBZHprdXFoZkU', fixations_file)
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

    return stimuli_val, stimuli_val


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
