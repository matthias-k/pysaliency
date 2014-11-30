from __future__ import absolute_import, print_function, division

from six.moves import urllib
import zipfile
import os
import shutil
import warnings
import hashlib

import numpy as np
from scipy.io import loadmat

from .datasets import FileStimuli, Stimuli, Fixations
from .utils import TemporaryDirectory


def check_file_hash(filename, md5_hash):
    """
    Check a file's hash and issue a warning it is has not the expected value.
    """
    print('Checking md5 sum...')
    with open(filename, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    if file_hash != md5_hash:
        warnings.warn("MD5 sum of {} has changed. Expected {} but got {}. This might lead to"
                      " this code producing wrong data.".format(filename, md5_hash, file_hash))


def download_file(url, target):
    """Download url to target while displaying progress information."""
    def log(blocks_recieved, block_size, file_size):
        print('\rDownloading file. {}% done'.format(int(blocks_recieved * block_size / file_size * 100)), end='')
    urllib.request.urlretrieve(url, target, log)
    print('')


def download_and_check(url, target, md5_hash):
    """Download url to target and check for correct md5_hash. Prints warning if hash is not correct."""
    download_file(url, target)
    check_file_hash(target, md5_hash)


def create_memory_stimuli(filenames):
    """
    Create a `Stimuli`-class from a list of filenames by reading the them
    """
    tmp_stimuli = FileStimuli(filenames)
    stimuli = list(tmp_stimuli.stimuli)  # Read all stimuli
    return Stimuli(stimuli)


def create_stimuli(stimuli_location, filenames, location=None):
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

        return FileStimuli(filenames)

    else:
        filenames = [os.path.join(stimuli_location, f) for f in filenames]
        return create_memory_stimuli(filenames)


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
        At the moment, the subjects stated in the Fixations object
        will not be correct, as they are difficult to infer from the
        published data (the data per subject is not stated in image dimensions)

    .. seealso::

        Neil Bruce, John K. Tsotsos. Attention based on information maximization [JoV 2007]

        `http://www-sop.inria.fr/members/Neil.Bruce/#SOURCECODE
    """
    if location:
        location = os.path.join(location, 'toronto')
        os.makedirs(location)
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
        fixations = Fixations.from_fixation_trains(xs, ys, ts, ns, subjects)

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
    @return: Stimuli, Fixations

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
                content = open(os.path.join(raw_path, subject, imagename+'.fix')).read()
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

        fixations = Fixations.from_fixation_trains(train_xs, train_ys, train_ts, train_ns, train_subjects)

    return stimuli, fixations
