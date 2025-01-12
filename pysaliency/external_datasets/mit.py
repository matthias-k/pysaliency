from __future__ import absolute_import, division, print_function

import glob
import os
import zipfile
from tempfile import TemporaryDirectory

import numpy as np
from natsort import natsorted
from PIL import Image
from pkg_resources import resource_string
from scipy.io import loadmat

from ..datasets import ScanpathFixations, Scanpaths
from ..utils import atomic_directory_setup, download_and_check, filter_files, run_matlab_cmd, get_matlab_or_octave
from .utils import _load, create_stimuli


def _get_mit1003(dataset_name, location=None, include_initial_fixation=False, only_1024_by_768=False, replace_initial_invalid_fixations=False):
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
        with TemporaryDirectory() as temp_dir:
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


            print("Patching checkFixations.m to work with octave")
            check_fixation_file = os.path.join(temp_dir, 'DatabaseCode', 'checkFixations.m')
            with open(check_fixation_file) as f:
                check_fixation_code = f.read()

            # nanmedian is not available in new octave statistics packages and deprecated in matlab
            # we replace it with median(data, 'omitnan') as suggested in the matlab documentation
            check_fixation_code = check_fixation_code.replace(
                'Fix.medianXY(1,1) = nanmedian(data(:,1));  Fix.medianXY(1,2) = nanmedian(data(:,2));',
                'Fix.medianXY(1,1) = median(data(:,1), "omitnan");  Fix.medianXY(1,2) = median(data(:,2), "omitnan");'
            )

            # same for nanvar
            check_fixation_code = check_fixation_code.replace(
                'Fix.driftXY(1,1) = nanvar(data(:,1));',
                'Fix.driftXY(1,1) = var(data(:,1), "omitnan");'
            )

            check_fixation_code = check_fixation_code.replace(
                'Fix.driftXY(1,2) = nanvar(data(:,2));',
                'Fix.driftXY(1,2) = var(data(:,2), "omitnan");'
            )

            with open(check_fixation_file, 'w') as f:
                f.write(check_fixation_code)

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
                    cmds.append("fprintf('Processing scanpath %d/%d\\r', {}, {});".format(n * len(subjects) + subject_id, total_cmd_count))
                    cmds.append("extract_fixations('{0}', '{1}', '{2}');".format(stimulus, subject_path, outfile))

            print('Running original code to extract fixations. This can take some minutes.')
            print('Warning: In the IPython Notebook, the output is shown on the console instead of the notebook.')
            with open(os.path.join(temp_dir, 'extract_all_fixations.m'), 'w') as f:
                for cmd in cmds:
                    f.write('{}\n'.format(cmd))

            command = 'extract_all_fixations;'

            is_octave = 'octave' in get_matlab_or_octave().lower()
            if is_octave:
                # in octave, the statistics package has to be loaded explicitly
                command = "pkg load statistics; {}".format(command)

            run_matlab_cmd(command, cwd=temp_dir)
            xs = []
            ys = []
            ts = []
            ns = []
            train_subjects = []
            duration_hist = []
            train_durations = []
            for n, stimulus in enumerate(stimuli_filenames):
                stimulus_size = stimuli.sizes[n]
                height, width = stimulus_size

                for subject_id, subject in enumerate(subjects):
                    subject_name = os.path.split(subject)[-1]
                    outfile = '{0}_{1}.mat'.format(stimulus, subject_name)
                    mat_data = loadmat(os.path.join(temp_dir, out_path, outfile))
                    fix_data = mat_data['fixations']
                    starts = mat_data['starts']
                    _durations = mat_data['durations']

                    _xs = mat_data['fixations'][:, 0]
                    _ys = mat_data['fixations'][:, 1]
                    _ts = mat_data['starts'].flatten()
                    _durations = mat_data['durations'].flatten()
                    full_durations = _durations.copy()

                    # this would be consistent with how it was handled in matlab
                    # however, originally I ported this slightly differnt
                    # and now I'm staying consistent withing pysaliency
                    # valid_indices = (
                    #    (np.floor(_xs) > 0) & (np.floor(_xs) < width)
                    #    & (np.floor(_ys) > 0) & (np.floor(_ys) < height))
                    valid_indices = (
                        (_xs > 0) & (_xs < width)
                        & (_ys > 0) & (_ys < height))

                    _xs = _xs[valid_indices]
                    _ys = _ys[valid_indices]
                    _ts = _ts[valid_indices]
                    _durations = _durations[valid_indices]

                    #_xs = _xs - 1  # matlab uses one-based indexing
                    #_ys = _ys - 1  # matlab uses one-based indexing
                    #_ts = _ts - 1  # data starts with t == 1, we want to use 0
                    _ts = _ts / 240.0  # Eye Tracker rate = 240Hz
                    _durations = _durations / 1000  # data is in ms, we want seconds

                    if not valid_indices[0] and (replace_initial_invalid_fixations or first_fixation > 0):
                        # if first fixation is invalid, no valid initial fixation is removed in the dataset
                        # for those cases, we add a central fixation with same duration as the invalid fixation
                        _xs = np.hstack(([0.5 * width], _xs))
                        _ys = np.hstack(([0.5 * height], _ys))
                        _ts = np.hstack(([0], _ts))
                        _durations = np.hstack(([full_durations[0] / 1000], _durations))

                    # if first_fixation == 1 the first fixation is skipped, as done
                    # by Judd.
                    _xs = _xs[first_fixation:]
                    _ys = _ys[first_fixation:]
                    _ts = _ts[first_fixation:]
                    _durations = _durations[first_fixation:]

                    xs.append(_xs)
                    ys.append(_ys)
                    ts.append(_ts)
                    ns.append(n)
                    train_subjects.append(subject_id)
                    train_durations.append(_durations)

                    for i in range(len(_durations)):
                        duration_hist.append(_durations[:i])

                    # x = []
                    # y = []
                    # t = []
                    # duration = []
                    # # TODO: This contains a subtle inconsistency: if the first fixation is invalid,
                    # # the next fixation is _not_ skipped. Therefore, the dataset still
                    # # contains a few "real" initial fixations. However, this is consistent
                    # # with how the dataset has been used in the past, so we are
                    # # keeping it.
                    # for i in range(first_fixation, fix_data.shape[0]):
                    #     if fix_data[i, 0] < 0 or fix_data[i, 1] < 0:
                    #         continue
                    #     if fix_data[i, 0] >= stimulus_size[1] or fix_data[i, 1] >= stimulus_size[0]:
                    #         continue
                    #     x.append(fix_data[i, 0])
                    #     y.append(fix_data[i, 1])
                    #     t.append(starts[0, i] / 240.0)  # Eye Tracker rate = 240Hz
                    #     duration_hist.append(np.array(duration))
                    #     duration.append(_durations[0, i] / 1000)  # data is in ms, we want seconds
                    # xs.append(x)
                    # ys.append(y)
                    # ts.append(t)
                    # ns.append(n)
                    # train_subjects.append(subject_id)
                    # train_durations.append(duration)

            #attributes = {
            #    # duration_hist contains for each fixation the durations of the previous fixations in the scanpath
            #    'duration_hist': build_padded_2d_array(duration_hist),
            #}
            #scanpath_attributes = {
            #    # train_durations contains the fixation durations for each scanpath
            #    'train_durations': build_padded_2d_array(train_durations),
            #}

            #scanpath_fixation_attributes = {
            #    'durations': train_durations,
            #}

            fixations = ScanpathFixations(Scanpaths(
                xs=xs,
                ys=ys,
                ts=ts,
                n=ns,
                subject=train_subjects,
                durations=train_durations,
            ))

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
        This code needs a working matlab installation including the Statistics and Machinelearning Toolbox as the original
        matlab code by Judd et al. is used to extract the fixation from the
        eyetracking data.

        Alternatively, the code should work with octave if the statistics package is in installed.

        The first fixation of each fixation train is discarded as stated in the
        paper (Judd et al. 2009).

    .. seealso::

        Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict where Humans Look [ICCV 2009]

        http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
    """
    return _get_mit1003('MIT1003', location=location, include_initial_fixation=False)


def get_mit1003_with_initial_fixation(location=None, replace_initial_invalid_fixations=False):
    """
    Loads or downloads and caches the MIT1003 dataset. The dataset
    consists of 1003 natural indoor and outdoor scenes of
    sizes: max dim: 1024px, other dim: 405-1024px
    and the fixations of 15 subjects under
    free viewing conditions with 3 seconds presentation time.

    All fixations outside of the image are discarded. This includes
    blinks.

    This version of the dataset include the initial central forced fixation,
    which is usually discarded. However, for scanpath prediction,
    it's important.

    Sometimes, the first recorded fixation is invalid. In this case,
    if `replace_initial_invalid_fixations` is True, it is replaced
    with a central fixation of the same length. This makes
    the dataset consistent with the ones without initial fixation
    in the sense of `fixations_without_initial_fixations = fixations_with[fixations_with.lengths > 0].

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
    name = 'MIT1003_initial_fix'
    if replace_initial_invalid_fixations:
        name += '_consistent'

    return _get_mit1003(name, location=location, include_initial_fixation=True, replace_initial_invalid_fixations=replace_initial_invalid_fixations)


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
        with TemporaryDirectory() as temp_dir:
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
