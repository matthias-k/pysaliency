from __future__ import absolute_import, print_function, division

import zipfile
import os
import shutil
import warnings
import glob

import numpy as np
from scipy.io import loadmat
from natsort import natsorted
import dill
from pkg_resources import resource_string
from PIL import Image

from .datasets import FileStimuli, Stimuli, FixationTrains
from .utils import TemporaryDirectory, filter_files, run_matlab_cmd, download_and_check


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
            stimuli = dill.load(open(os.path.join(location, 'stimuli.pydat'), 'rb'))
            fixations = dill.load(open(os.path.join(location, 'fixations.pydat'), 'rb'))
            return stimuli, fixations
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
        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, subjects)
    if location:
        dill.dump(stimuli, open(os.path.join(location, 'stimuli.pydat'), 'wb'))
        dill.dump(fixations, open(os.path.join(location, 'fixations.pydat'), 'wb'))
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

        fixations = FixationTrains.from_fixation_trains(train_xs, train_ys, train_ts, train_ns, train_subjects)

    if location:
        dill.dump(stimuli, open(os.path.join(location, 'stimuli.pydat'), 'wb'))
        dill.dump(fixations, open(os.path.join(location, 'fixations.pydat'), 'wb'))
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
    if location:
        location = os.path.join(location, 'MIT1003')
        if os.path.exists(location):
            stimuli = dill.load(open(os.path.join(location, 'stimuli.pydat'), 'rb'))
            fixations = dill.load(open(os.path.join(location, 'fixations.pydat'), 'rb'))
            return stimuli, fixations
        os.makedirs(location)
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
                cmds.append("fprintf('%d/%d\\n', {}, {});".format(n*len(subjects)+subject_id, total_cmd_count))
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
        for n, stimulus in enumerate(stimuli_filenames):
            stimulus_size = stimuli.sizes[n]
            for subject_id, subject in enumerate(subjects):
                subject_name = os.path.split(subject)[-1]
                outfile = '{0}_{1}.mat'.format(stimulus, subject_name)
                mat_data = loadmat(os.path.join(temp_dir, out_path, outfile))
                fix_data = mat_data['fixations']
                starts = mat_data['starts']
                x = []
                y = []
                t = []
                for i in range(1, fix_data.shape[0]):  # Skip first fixation like Judd does
                    if fix_data[i, 0] < 0 or fix_data[i, 1] < 0:
                        continue
                    if fix_data[i, 0] >= stimulus_size[1] or fix_data[i, 1] >= stimulus_size[0]:
                        continue
                    x.append(fix_data[i, 0])
                    y.append(fix_data[i, 1])
                    t.append(starts[0, i]/240.0)  # Eye Tracker rate = 240Hz
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ns.append(n)
                train_subjects.append(subject_id)
        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects)

    if location:
        dill.dump(stimuli, open(os.path.join(location, 'stimuli.pydat'), 'wb'))
        dill.dump(fixations, open(os.path.join(location, 'fixations.pydat'), 'wb'))
    return stimuli, fixations


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
    if location:
        location = os.path.join(location, 'MIT1003_onesize')
        if os.path.exists(location):
            stimuli = dill.load(open(os.path.join(location, 'stimuli.pydat'), 'rb'))
            fixations = dill.load(open(os.path.join(location, 'fixations.pydat'), 'rb'))
            return stimuli, fixations
        os.makedirs(location)
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
        stimuli_filenames = sorted(images)  # I used sorted instead of natsorted at this time

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

        subjects = ['kae', 'jw', 'tmj', 'ems', 'zb', 'emb', 'hp', 'ajs', 'tu', 'jcw', 'krl', 'CNG', 'ff', 'po', 'ya']

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
                cmds.append("fprintf('%d/%d\\n', {}, {});".format(n*len(subjects)+subject_id, total_cmd_count))
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
        for n, stimulus in enumerate(stimuli_filenames):
            stimulus_size = stimuli.sizes[n]
            for subject_id, subject in enumerate(subjects):
                subject_name = os.path.split(subject)[-1]
                outfile = '{0}_{1}.mat'.format(stimulus, subject_name)
                mat_data = loadmat(os.path.join(temp_dir, out_path, outfile))
                fix_data = mat_data['fixations']
                starts = mat_data['starts']
                x = []
                y = []
                t = []
                for i in range(1, fix_data.shape[0]):  # Skip first fixation like Judd does
                    if fix_data[i, 0] < 0 or fix_data[i, 1] < 0:
                        continue
                    if fix_data[i, 0] >= stimulus_size[1] or fix_data[i, 1] >= stimulus_size[0]:
                        continue
                    x.append(fix_data[i, 0])
                    y.append(fix_data[i, 1])
                    t.append(starts[0, i]/240.0)  # Eye Tracker rate = 240Hz
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ns.append(n)
                train_subjects.append(subject_id)
        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects)

    if location:
        dill.dump(stimuli, open(os.path.join(location, 'stimuli.pydat'), 'wb'))
        dill.dump(fixations, open(os.path.join(location, 'fixations.pydat'), 'wb'))
    return stimuli, fixations


def get_cat2000_train(location=None):
    """
    Loads or downloads and caches the CAT2000 dataset. The dataset
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
    if location:
        location = os.path.join(location, 'CAT2000_train')
        if os.path.exists(location):
            stimuli = dill.load(open(os.path.join(location, 'stimuli.pydat'), 'rb'))
            fixations = dill.load(open(os.path.join(location, 'fixations.pydat'), 'rb'))
            return stimuli, fixations
        os.makedirs(location)
    with TemporaryDirectory(cleanup=True) as temp_dir:
        download_and_check('http://saliency.mit.edu/trainSet.zip',
                           os.path.join(temp_dir, 'trainSet.zip'),
                           '0d7df8b954ecba69b6796e77b9afe4b6')

        # Stimuli
        print('Creating stimuli')
        f = zipfile.ZipFile(os.path.join(temp_dir, 'trainSet.zip'))
        f.extractall(temp_dir)

        stimuli_src_location = os.path.join(temp_dir, 'Stimuli')
        stimuli_target_location = os.path.join(location, 'stimuli') if location else None
        images = glob.glob(os.path.join(stimuli_src_location, '*', '*.jpeg'))
        images = [os.path.relpath(img, start=stimuli_src_location) for img in images]
        stimuli_filenames = natsorted(images)

        stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

        # FixationTrains

        print('Creating fixations')

        with open(os.path.join(temp_dir, 'load_cat2000.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/{}'.format('load_cat2000.m')))

        cmds = []
        # It is vital _not_ to store the extracted fixations in the main
        # directory where matlab is running, as matlab will check the timestamp
        # of all files in this directory very often. This leads to heavy
        # performance penalties and would make matlab run for more than an
        # hour.
        out_path = 'extracted'
        os.makedirs(os.path.join(temp_dir, out_path))
        run_matlab_cmd('load_cat2000;', cwd=temp_dir)

        print('Running original code to extract fixations. This can take some minutes.')
        print('Warning: In the IPython Notebook, the output is shown on the console instead of the notebook.')

        run_matlab_cmd('extract_all_fixations;', cwd=temp_dir)
        xs = []
        ys = []
        ts = []
        ns = []
        train_subjects = []
        for n, stimulus in enumerate(stimuli_filenames):
            stimulus_size = stimuli.sizes[n]
            for subject_id, subject in enumerate(subjects):
                subject_name = os.path.split(subject)[-1]
                outfile = '{0}_{1}.mat'.format(stimulus, subject_name)
                mat_data = loadmat(os.path.join(temp_dir, out_path, outfile))
                fix_data = mat_data['fixations']
                starts = mat_data['starts']
                x = []
                y = []
                t = []
                for i in range(1, fix_data.shape[0]):  # Skip first fixation like Judd does
                    if fix_data[i, 0] < 0 or fix_data[i, 1] < 0:
                        continue
                    if fix_data[i, 0] >= stimulus_size[1] or fix_data[i, 1] >= stimulus_size[0]:
                        continue
                    x.append(fix_data[i, 0])
                    y.append(fix_data[i, 1])
                    t.append(starts[0, i]/240.0)  # Eye Tracker rate = 240Hz
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ns.append(n)
                train_subjects.append(subject_id)
        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects)

    if location:
        dill.dump(stimuli, open(os.path.join(location, 'stimuli.pydat'), 'wb'))
        dill.dump(fixations, open(os.path.join(location, 'fixations.pydat'), 'wb'))
    return stimuli, fixations
