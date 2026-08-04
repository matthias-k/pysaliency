import os
import shutil
import struct
import zipfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import piexif
from tqdm import tqdm

from ..datasets import ScanpathFixations, Scanpaths, create_subset
from ..utils import (
    atomic_directory_setup,
    download_and_check,
    filter_files,
    get_minimal_unique_filenames,
)
from .utils import _load, create_stimuli

# The stimuli live in the public OSF component "DAEMONS Image Data Set" (osf.io/cn5yp).
# We fetch the two corpus archives by their individual file URLs rather than as a zip of the
# containing folder: the folder bundle is generated on the fly, so its hash is not guaranteed
# stable, and it also carries ~160 MB of thumbnail PDFs and CSVs that we do not use.
IMAGE_ARCHIVES = {
    'DAEMONS_flickr_corpus.zip': ('https://osf.io/download/hkvj5/', 'a5720326e3331800d0921767b7e23716'),
    'DAEMONS_potsdam_corpus.zip': ('https://osf.io/download/x5jb2/', '17d72179b8ab3552e8ada8c0a1a2a64e'),
}

# The scanpaths live in the public OSF component "DAEMONS Eye Tracking Data" (osf.io/2ux87).
# It contains the training and validation saccade tables only; the test split is withheld for
# use in a benchmark and has to be supplied via `test_data`.
FIXATION_ARCHIVE = ('eye_movement.zip', 'https://osf.io/download/ztgna/', '2779b4c140a0b1e3c9976488994f08f3')

# Filenames inside the zip archives that are not stimuli.
NON_STIMULUS_FILES = ['__MACOSX', '.DS_Store', 'potsdam_corpus_resz.sh']

# All stimuli are presented at this size, and the screen spans 32 degrees of visual angle
# horizontally, giving 60 px/dva.
STIMULUS_SIZE = (1080, 1920)
SCREEN_WIDTH_DEGREES = 32

# Indices of the test stimuli within the full, sorted list of all 2400 stimuli. The test
# scanpaths are withheld, so without this the test stimuli could not be identified from public
# data at all. The three splits partition the full set: 2000 + 200 + 200 = 2400.
TEST_STIMULUS_INDICES = [
    32, 37, 66, 72, 86, 107, 116, 130, 132, 159,
    160, 162, 163, 168, 191, 195, 216, 231, 234, 243,
    246, 279, 282, 287, 288, 293, 295, 299, 303, 309,
    334, 345, 371, 378, 401, 414, 443, 452, 476, 489,
    492, 517, 547, 567, 570, 573, 595, 602, 640, 643,
    646, 649, 661, 664, 671, 680, 787, 798, 801, 808,
    813, 833, 835, 836, 849, 864, 865, 866, 869, 885,
    895, 906, 921, 925, 933, 946, 954, 959, 967, 968,
    994, 1005, 1018, 1020, 1032, 1058, 1099, 1112, 1126, 1141,
    1150, 1157, 1162, 1169, 1171, 1204, 1211, 1213, 1218, 1241,
    1251, 1263, 1275, 1328, 1333, 1346, 1355, 1390, 1414, 1415,
    1436, 1441, 1460, 1461, 1462, 1463, 1464, 1485, 1524, 1525,
    1527, 1535, 1544, 1554, 1560, 1578, 1599, 1614, 1617, 1636,
    1645, 1668, 1685, 1718, 1742, 1745, 1756, 1758, 1783, 1785,
    1804, 1805, 1819, 1828, 1846, 1862, 1864, 1895, 1919, 1922,
    1931, 1948, 1957, 1966, 1968, 2004, 2008, 2016, 2022, 2024,
    2034, 2042, 2056, 2065, 2067, 2071, 2074, 2078, 2099, 2110,
    2114, 2115, 2148, 2149, 2150, 2155, 2177, 2191, 2192, 2202,
    2204, 2207, 2209, 2217, 2222, 2225, 2235, 2242, 2261, 2265,
    2269, 2291, 2298, 2321, 2322, 2334, 2376, 2377, 2382, 2384,
]


def _download_DAEMONS(image_directory, fixation_directory):
    """Download the DAEMONS source archives.

    The image corpora go to `image_directory` and the public saccade tables to
    `fixation_directory`; they are separate because the fixation archive is small enough to keep
    alongside the built dataset while the ~2.5 GB of images is not.
    """
    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(fixation_directory, exist_ok=True)

    for filename, (url, md5_hash) in IMAGE_ARCHIVES.items():
        download_and_check(url, os.path.join(image_directory, filename), md5_hash)

    filename, url, md5_hash = FIXATION_ARCHIVE
    download_and_check(url, os.path.join(fixation_directory, filename), md5_hash)

    return os.path.join(fixation_directory, filename)


def _create_DAEMONS_stimuli(image_directory, stimuli_target_location=None):
    """Extract the image corpora and build the full stimulus collection.

    Returns all 2400 stimuli in sorted filename order, which is the order the scanpath image
    indices refer to.
    """
    with TemporaryDirectory() as temp_dir:
        stimuli_src_location = os.path.join(temp_dir, 'stimuli')
        os.makedirs(stimuli_src_location)

        extracted = []
        for filename in IMAGE_ARCHIVES:
            with zipfile.ZipFile(os.path.join(image_directory, filename)) as archive:
                namelist = filter_files(archive.namelist(), NON_STIMULUS_FILES)
                archive.extractall(stimuli_src_location, namelist)
                extracted.extend(namelist)

        # the archives contain one directory per corpus; flatten them. A basename shared between
        # the two corpora would silently overwrite, so rule that out rather than trust it.
        basenames = [os.path.basename(name) for name in extracted if name.endswith('.jpg')]
        if len(basenames) != len(set(basenames)):
            raise ValueError("the image corpora contain colliding filenames")

        print("Flattening corpus directories...")
        for name in tqdm(extracted):
            path = os.path.join(stimuli_src_location, name)
            if os.path.isfile(path):
                shutil.move(path, os.path.join(stimuli_src_location, os.path.basename(name)))
        for name in os.listdir(stimuli_src_location):
            path = os.path.join(stimuli_src_location, name)
            if os.path.isdir(path):
                shutil.rmtree(path)

        filenames = sorted(
            os.path.basename(name) for name in extracted if name.endswith('.jpg')
        )

        # Some images carry EXIF orientation tags, which PIL and imageio ignore but image viewers
        # honour. The scanpaths show that every image was presented unrotated, so we normalise the
        # tag rather than the pixels.
        print("Normalizing EXIF orientation...")
        for filename in tqdm(filenames):
            _normalize_exif_orientation(os.path.join(stimuli_src_location, filename))

        stimuli = create_stimuli(stimuli_src_location, filenames, stimuli_target_location)

    if not set(stimuli.sizes) == {STIMULUS_SIZE}:
        raise ValueError(f"Unexpected image sizes: {set(stimuli.sizes) - {STIMULUS_SIZE}}")

    return stimuli


def _normalize_exif_orientation(filename):
    try:
        exif_dict = piexif.load(filename)
    except struct.error as e:
        print(f"Failed to load EXIF data of {filename}, {e}")
        return

    orientation = exif_dict['0th'].get(piexif.ImageIFD.Orientation)
    if orientation is not None and orientation != 1:
        exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
        piexif.insert(piexif.dump(exif_dict), filename)


def _load_scanpaths(df, stimuli):
    """Build scanpaths from one DAEMONS saccade table.

    Each row of the table is a saccade and the fixation it leads to. Rows without fixation
    coordinates are dropped: they occur when the recording ended mid-saccade, and in the
    non-public version of the data also at the start of some trials.

    `mask_shown` and `forced_end` are only present in the public release and are added as
    scanpath attributes when the columns exist.
    """
    stimulus_indices = {
        filename: index
        for index, filename in enumerate(get_minimal_unique_filenames(stimuli.filenames))
    }
    height, width = STIMULUS_SIZE
    px_per_dva = width / SCREEN_WIDTH_DEGREES

    df = df.copy()
    # A trial is a maximal run of rows with the same subject and trial number. Numbering the runs
    # with a cumulative sum makes the index monotonically increasing, so grouping by it sorted
    # yields the scanpaths in the order the rows appear in the table.
    df['scanpath_index'] = (
        (df['trial'] != df['trial'].shift(1)) | (df['VP'] != df['VP'].shift(1))
    ).cumsum()

    valid = df['x'].notna()
    print(f"Skipping {(~valid).sum()} fixations without coordinates")
    df = df[valid]

    scanpaths = df.groupby('scanpath_index', sort=True)

    xs = []
    ys = []
    ts = []
    ns = []
    subjects = []
    durations = []
    blinks = []
    sticky = []
    forced_fixations = []
    trials = []
    mask_shown = []
    forced_end = []

    has_mask_shown = 'mask_shown' in df.columns
    has_forced_end = 'forced_end' in df.columns

    for _, rows in tqdm(scanpaths, total=len(scanpaths)):
        first = rows.iloc[0]

        xs.append((rows['x'] * px_per_dva).to_numpy())
        # the saccade tables use a bottom-left origin, pysaliency a top-left one
        ys.append((height - rows['y'] * px_per_dva).to_numpy())
        # `End` is the end of the saccade, i.e. the onset of the fixation. It is missing for the
        # first fixation of a trial, which has no preceding saccade.
        ts.append((rows['End'].fillna(0.0) / 1000).to_numpy())
        ns.append(stimulus_indices[first['Img']])
        subjects.append(first['VP'])
        durations.append((rows['fixdur'] / 1000).to_numpy(dtype=float))
        # The flag columns arrive with inconsistent dtypes. `blinkFix` is read as object because
        # it is missing wherever the fixation itself is missing, and pandas has no NA-capable bool
        # in the classic dtypes, so the column degrades to Python bools plus float NaN. `sticky` is
        # float in the non-public tables but int in the public ones. Cast: `VariableLengthArray`
        # pads with NaN and so has to be float anyway, and an object array cannot be written to
        # HDF5 at all.
        blinks.append(rows['blinkFix'].to_numpy(dtype=float))
        sticky.append(rows['sticky'].to_numpy(dtype=float))
        forced_fixations.append(rows['forced_fix'].to_numpy(dtype=float))
        trials.append(first['trial'])
        if has_mask_shown:
            mask_shown.append(bool(first['mask_shown']))
        if has_forced_end:
            forced_end.append(first['forced_end'] / 1000)

    scanpath_attributes = {'trial': trials}
    if has_mask_shown:
        scanpath_attributes['mask_shown'] = mask_shown
    if has_forced_end:
        scanpath_attributes['forced_end'] = forced_end

    return ScanpathFixations(Scanpaths(
        xs=xs,
        ys=ys,
        ts=ts,
        n=ns,
        subject=subjects,
        scanpath_attributes=scanpath_attributes,
        fixation_attributes={
            'durations': durations,
            'blinks': blinks,
            'sticky': sticky,
            'forced_fixations': forced_fixations,
        },
        attribute_mapping={
            'durations': 'duration',
            'blinks': 'blink',
        },
    ))


def _read_saccade_table(archive_filename, split):
    """Read one saccade table from a DAEMONS scanpath archive.

    The tables are looked up by filename anywhere in the archive rather than at a fixed path, since
    the public release and the non-public one wrap them in differently named directories.
    """
    with zipfile.ZipFile(archive_filename) as archive:
        candidates = [
            name for name in archive.namelist()
            if os.path.basename(name) == f'SAC_{split}.csv' and '__MACOSX' not in name
        ]
        if not candidates:
            raise ValueError(f"{archive_filename} does not contain SAC_{split}.csv")
        with archive.open(candidates[0]) as f:
            return pd.read_csv(f)


def get_DAEMONS(location=None, test_data=None):
    """
    Loads or downloads and caches the DAEMONS dataset.

    DAEMONS (Potsdam Dataset for Eye Movement On Natural Scenes) consists of 2400 images of
    1920x1080 pixels, viewed by 250 observers in a freeviewing task. Each trial starts with a
    forced fixation on a marker, which disappears when the stimulus is shown.

    The scanpaths come with fixation attributes for
    - (fixation) duration in seconds
    - blink: whether a blink occurred during the fixation
    - sticky: carried over from the source data, which does not document its meaning
    - forced_fixations: whether the fixation happened during the forced fixation period.
      Note that about 8% of trials have more than one such fixation, because observers make
      small saccades within the forced fixation period. Code that treats the forced fixation as
      the initial condition of a scanpath should use this attribute rather than assuming that
      exactly the first fixation is forced.

    and scanpath attributes for
    - trial: the trial number within the observer's session
    - mask_shown: whether the initial fixation check had to be restarted for this trial
    - forced_end: the time in seconds at which the forced fixation period ended

    The test split scanpaths are withheld for use in a benchmark and are not part of the public
    data. The test stimuli are returned regardless. If you have access to the full saccade
    tables, pass them as `test_data` and the test scanpaths will additionally be built and saved
    (but not returned, since the split is meant to stay held out).

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset will be stored in the
                     subdirectory `DAEMONS` of location and read from there, if already present.
                     The downloaded scanpath archive is kept in `DAEMONS/src`, together with a
                     copy of `test_data` if one was given.
    @type  test_data: string, defaults to `None`
    @param test_data: filename of an archive containing `SAC_test.csv`, if you have access to
                      the non-public data. A copy of this file is placed in `DAEMONS/src`.

    @return: Training stimuli, training scanpaths, validation stimuli, validation scanpaths,
             test stimuli

    .. seealso::

        Schwetlick, L., Backhaus, D., & Engbert, R. (2022). A dynamical scan-path model for
        task-dependence during scene viewing. Psychological Review.

        https://osf.io/ewr5u/ (project) - https://osf.io/cn5yp/ (images) -
        https://osf.io/2ux87/ (eye tracking data)
    """
    if location:
        location = os.path.join(location, 'DAEMONS')
        if os.path.exists(location):
            stimuli_train = _load(os.path.join(location, 'stimuli_train.hdf5'))
            scanpaths_train = _load(os.path.join(location, 'fixations_train.hdf5'))
            stimuli_validation = _load(os.path.join(location, 'stimuli_validation.hdf5'))
            scanpaths_validation = _load(os.path.join(location, 'fixations_validation.hdf5'))
            stimuli_test = _load(os.path.join(location, 'stimuli_test.hdf5'))

            return stimuli_train, scanpaths_train, stimuli_validation, scanpaths_validation, stimuli_test
        os.makedirs(location)

    with atomic_directory_setup(location):
        with TemporaryDirectory() as temp_dir:
            source_directory = os.path.join(location, 'src') if location else temp_dir

            fixation_archive = _download_DAEMONS(temp_dir, source_directory)

            if test_data:
                test_data_copy = os.path.join(source_directory, os.path.basename(test_data))
                if os.path.abspath(test_data) != os.path.abspath(test_data_copy):
                    shutil.copyfile(test_data, test_data_copy)

            print("Creating stimuli...")
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            stimuli = _create_DAEMONS_stimuli(temp_dir, stimuli_target_location)

            print("Creating training scanpaths...")
            scanpaths_train_all = _load_scanpaths(_read_saccade_table(fixation_archive, 'train'), stimuli)
            print("Creating validation scanpaths...")
            scanpaths_validation_all = _load_scanpaths(_read_saccade_table(fixation_archive, 'val'), stimuli)

            stimuli_train, scanpaths_train = create_subset(
                stimuli, scanpaths_train_all, sorted(set(scanpaths_train_all.n)))
            stimuli_validation, scanpaths_validation = create_subset(
                stimuli, scanpaths_validation_all, sorted(set(scanpaths_validation_all.n)))

            if test_data:
                print("Creating test scanpaths...")
                scanpaths_test_all = _load_scanpaths(_read_saccade_table(test_data, 'test'), stimuli)
                ns_test = sorted(set(scanpaths_test_all.n))

                if len(ns_test) != len(TEST_STIMULUS_INDICES) or not np.all(
                        np.array(ns_test) == TEST_STIMULUS_INDICES):
                    raise ValueError(
                        f"the test scanpaths cover {len(ns_test)} stimuli, which does not match "
                        f"the {len(TEST_STIMULUS_INDICES)} expected test stimuli"
                    )

                _, scanpaths_test = create_subset(stimuli, scanpaths_test_all, ns_test)

            stimuli_test = stimuli[TEST_STIMULUS_INDICES]

        if location:
            stimuli_train.to_hdf5(os.path.join(location, 'stimuli_train.hdf5'))
            scanpaths_train.to_hdf5(os.path.join(location, 'fixations_train.hdf5'))
            stimuli_validation.to_hdf5(os.path.join(location, 'stimuli_validation.hdf5'))
            scanpaths_validation.to_hdf5(os.path.join(location, 'fixations_validation.hdf5'))
            stimuli_test.to_hdf5(os.path.join(location, 'stimuli_test.hdf5'))
            if test_data:
                scanpaths_test.to_hdf5(os.path.join(location, 'fixations_test.hdf5'))

    return stimuli_train, scanpaths_train, stimuli_validation, scanpaths_validation, stimuli_test


def get_DAEMONS_train(location=None):
    stimuli_train, scanpaths_train, stimuli_validation, scanpaths_validation, stimuli_test = get_DAEMONS(location=location)
    return stimuli_train, scanpaths_train


def get_DAEMONS_validation(location=None):
    stimuli_train, scanpaths_train, stimuli_validation, scanpaths_validation, stimuli_test = get_DAEMONS(location=location)
    return stimuli_validation, scanpaths_validation


def get_DAEMONS_test(location=None):
    stimuli_train, scanpaths_train, stimuli_validation, scanpaths_validation, stimuli_test = get_DAEMONS(location=location)
    return stimuli_test
