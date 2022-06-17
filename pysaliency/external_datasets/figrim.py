from __future__ import absolute_import, print_function, division

import zipfile
import os
import glob

import numpy as np
from scipy.io import loadmat
from natsort import natsorted
from boltons.fileutils import mkdir_p

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    download_and_check,
    atomic_directory_setup,
)

from .utils import create_stimuli, _load


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
                scanpath_attributes={
                    'which_time': which_times,
                    'stimulus_type': stimulus_types,
                    'response': responses
                })

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations