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

            source_directory = os.path.join(location, 'src')
            os.makedirs(source_directory)

            source_file = os.path.join(source_directory, 'NUSEF_database.zip')

            download_and_check('https://ncript.comp.nus.edu.sg/site/mmas/NUSEF_database.zip',
                               source_file,
                               '429a78ad92184e8a4b37419988d98953')

            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(source_file)
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

            fix_location = os.path.join(temp_dir, 'NUSEF_database', 'fix_data')
            for sub_dir in tqdm(os.listdir(fix_location)):
                if sub_dir + '.jpg' not in stimuli_indices:
                    # one of the non public images
                    continue
                n = stimuli_indices[sub_dir + '.jpg']

                scale_x = 1024 / 260
                scale_y = 768 / 280

                size = stimuli.sizes[n]

                # according to the MATLAB visualiation code, images were scaled to screen size by
                # 1. scaling the images to have a height of 768 pixels
                # 2. checking if the resulting width is larger than 1024, in this case
                #    the image is downscaled to have a width of 1024
                #    (and hence a height of less than 768)
                # here we recompute the scale factors so that we can compute fixation locations
                # in image coordinates from the screen coordinates
                image_resize_factor = 768 / size[0]
                resized_height = 768
                resized_width = size[1] * image_resize_factor
                if resized_width > 1024:
                    image_resize_factor * (1024 / resized_width)
                    resized_width = 1024
                    resized_height *= (1024 / resized_width)

                # images were shown centered
                x_offset = (1024 - resized_width) / 2
                y_offset = (768 - resized_height) / 2

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

                        # transform from eye trackoer to screen pixels
                        this_x = float(hor_pos) * scale_x
                        this_y = float(ver_pos) * scale_y

                        # transform to screen image coordinate
                        this_x -= x_offset
                        this_y -= y_offset

                        # transform to original image coordinates
                        this_x /= image_resize_factor
                        this_y /= image_resize_factor

                        x.append(this_x)
                        y.append(this_y)

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

        fixations = FixationTrains.from_fixation_trains(
            xs,
            ys,
            ts,
            ns,
            train_subjects,
            scanpath_fixation_attributes={
            'durations': durations,
            },
            scanpath_attribute_mapping={'durations': 'duration'}
        )

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations
