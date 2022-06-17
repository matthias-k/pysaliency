from __future__ import absolute_import, print_function, division

import os
import urllib

import numpy as np
from scipy.io import loadmat
from boltons.fileutils import mkdir_p
from tqdm import tqdm

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    download_and_check,
    atomic_directory_setup,
)

from .utils import create_stimuli, _load


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