from __future__ import absolute_import, division, print_function

import glob
import os
import zipfile
from datetime import datetime, timedelta

from tqdm import tqdm

from ..datasets import FixationTrains
from ..utils import (
    TemporaryDirectory,
    atomic_directory_setup,
    download_and_check,
)
from .utils import _load, create_stimuli


# TODO: extract fixation durations
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

    Subjects ids used currently might not be the real subject ids 
    and might be inconsistent across images.

    The data collection experiment didn't enforce a specific 
    fixation at stimulus onset.

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `toronto` of
                     location and read from there, if already present.
    @return: Stimuli, FixationTrains

    .. seealso::

        Subramanian Ramanathan, Harish Katti, Nicu Sebe, Mohan Kankanhalli, Tat-Seng Chua. An eye fixation database for saliency detection in images [ECCV 2010]

        http://mmas.comp.nus.edu.sg/NUSEF.html (link seems outdated)

        https://ncript.comp.nus.edu.sg/site/mmas/NUSEF.html (seems to be the new location of the dataset)
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

            download_and_check('https://ncript.comp.nus.edu.sg/site/mmas/NUSEF_database.zip',
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
            durations = []
            date_format = "%H:%M:%S.%f"

            scale_x = 1024 / 260
            scale_y = 768 / 280

            fix_location = os.path.join(temp_dir, 'NUSEF_database', 'fix_data')
            for sub_dir in tqdm(os.listdir(fix_location)):
                if not sub_dir + '.jpg' in stimuli_indices:
                    # one of the non public images
                    continue
                n = stimuli_indices[sub_dir + '.jpg']
                for subject_data in glob.glob(os.path.join(fix_location, sub_dir, '*.fix')):
                    subject_id = int(subject_data.split('+')[0][-2:])
                    data = open(subject_data).read().replace('\r\n', '\n')
                    data = data.split('COLS=', 1)[1]
                    data = data.split('[Fix Segment Summary')[0]
                    lines = data.split('\n')[1:]
                    lines = [l for l in lines if l]
                    x = []
                    y = []
                    t = []
                    fixation_durations = []
                    initial_start_time = None
                    for i in range(len(lines)):
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
                         interfix_loss) = lines[i].split()
                        x.append(float(hor_pos) * scale_x)
                        y.append(float(ver_pos) * scale_y)
                        current_start_time = datetime.strptime(str(start_time), date_format)
                        if i == 0:
                            initial_start_time = current_start_time
                        t.append(float((current_start_time - initial_start_time).total_seconds()))
                        fixation_durations.append(float(fix_dur))

                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                    ns.append(n)
                    train_subjects.append(subject_id)
                    durations.append(fixation_durations)

        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects, durations)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations
