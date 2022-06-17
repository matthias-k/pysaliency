from __future__ import absolute_import, print_function, division

import zipfile
import os

import itertools
import numpy as np
from scipy.io import loadmat

from boltons.fileutils import mkdir_p
from tqdm import tqdm

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    atomic_directory_setup,
    check_file_hash,
    )

from .utils import  create_stimuli, _load

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
        #mkdir_p(location)
    if not datafile:
        raise ValueError('The Koehler dataset is not freely available! You have to '
                         'request the data file from the authors and provide it to '
                         'this function via the datafile argument')
    check_file_hash(datafile, '405f58aaa9b4ddc76f3e8f23c379d315')
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