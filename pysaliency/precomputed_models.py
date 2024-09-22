from __future__ import absolute_import, division, print_function

import glob
import logging
import os.path
import tarfile
import warnings
import zipfile

import numpy as np
from imageio import imread
from scipy.io import loadmat
from scipy.special import logsumexp
from tqdm import tqdm

from .datasets import FileStimuli, get_image_hash
from .models import Model
from .saliency_map_models import SaliencyMapModel
from .utils import full_split, get_minimal_unique_filenames


def get_stimuli_filenames(stimuli):
    if not isinstance(stimuli, FileStimuli) and 'filenames' not in stimuli.__attributes__:
        raise ValueError("Need FileStimuli or Stimuli with filenames attribute")

    if 'filenames' in stimuli.attributes:
        return stimuli.attributes['filenames']
    else:
        return stimuli.filenames


def get_keys_from_filenames(filenames, keys):
    """checks how much filenames have to be shorted to get the correct hdf5 or other keys"""
    if not filenames:
        return []

    first_filename_parts = full_split(filenames[0])
    for part_index in range(len(first_filename_parts)):
        remaining_filename = os.path.join(*first_filename_parts[part_index:])
        if remaining_filename in keys:
            break
    else:
        print("No common prefix found!")
        print(f"  filename: {filenames[0]}")
        print("  keys:")
        for key in keys[:5]:
            print(f"    {key}")
        for key in keys[-5:]:
            print(f"    {key}")

        raise ValueError('No common prefix found!')

    filename_keys = []
    for filename in filenames:
        filename_parts = full_split(filename)
        remaining_filename = os.path.join(*filename_parts[part_index:])
        filename_keys.append(remaining_filename)

    return filename_keys


def get_keys_from_filenames_with_prefix(filenames, keys):
    """checks how much filenames have to be shorted to get the correct hdf5 or other keys, where the keys might have a prefix"""
    first_key_parts = full_split(keys[0])

    for key_part_index in range(len(first_key_parts)):
        remaining_keys = [os.path.join(*full_split(key)[key_part_index:]) for key in keys]
        try:
            filename_keys = get_keys_from_filenames(filenames, remaining_keys)
        except ValueError:
            continue
        else:
            full_filename_keys = []
            for key, filename_key in zip(keys, filename_keys):
                full_filename_keys.append(os.path.join(*full_split(key)[:key_part_index], filename_key))
            return full_filename_keys

    raise ValueError('No common prefix found from {} and {}'.format(filenames[0], keys[0]))


def export_model_to_hdf5(model, stimuli, filename, compression=9, overwrite=True, flush=False):
    """Export pysaliency model predictions for stimuli into hdf5 file

    model: Model or SaliencyMapModel
    stimuli: instance of FileStimuli or Stimuli with filenames attribute
    filename: where to save hdf5 file to
    compression: how much to compress the data
    overwrite: if False, an existing file will be appended to and
      if for some stimuli predictions already exist, they will be
      kept.
    flush: whether the hdf5 file should be flushed after each stimulus
    """
    filenames = get_stimuli_filenames(stimuli)
    names = get_minimal_unique_filenames(filenames)

    import h5py

    if overwrite:
        mode = 'w'
    else:
        mode = 'a'

    with h5py.File(filename, mode=mode) as f:
        if overwrite:
            indices = range(len(stimuli))
        else:
            indices = [i for i in range(len(stimuli)) if names[i] not in f]
            logging.debug(f"Skipping {len(stimuli) - len(indices)} already existing entries")
        for k in tqdm(indices):
            stimulus = stimuli[k]

            if isinstance(model, SaliencyMapModel):
                smap = model.saliency_map(stimulus)
            elif isinstance(model, Model):
                smap = model.log_density(stimulus)
            else:
                raise TypeError(type(model))
            f.create_dataset(names[k], data=smap, compression=compression)
            if flush:
                f.flush()


class SaliencyMapModelFromFiles(SaliencyMapModel):
    def __init__(self, stimuli, files, **kwargs):
        super(SaliencyMapModelFromFiles, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.stimulus_ids = list(stimuli.stimulus_ids)
        self.files = files
        assert(len(files) == len(stimuli))

    def _saliency_map(self, stimulus):
        filename = self._file_for_stimulus(stimulus)
        return self._load_file(filename)

    def _file_for_stimulus(self, stimulus):
        stimulus_id = get_image_hash(stimulus)

        try:
            stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
        except IndexError as exc:
            raise IndexError("Stimulus id '{}' not found in stimuli!".format(stimulus_id)) from exc

        return self.files[stimulus_index]

    def _load_file(self, filename):
        _, ext = os.path.splitext(filename)
        if ext.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
            return imread(filename).astype(float)
        elif ext.lower() == '.npy':
            return np.load(filename).astype(float)
        elif ext.lower() == '.mat':
            data = loadmat(filename)
            variables = [v for v in data if not v.startswith('__')]
            if len(variables) > 1:
                raise ValueError('{} contains more than one variable: {}'.format(filename, variables))
            elif len(variables) == 0:
                raise ValueError('{} contains no data'.format(filename))
            return data[variables[0]]
        else:
            raise ValueError('Unkown file type: {}'.format(ext))


class SaliencyMapModelFromDirectory(SaliencyMapModelFromFiles):
    def __init__(self, stimuli, directory, **kwargs):
        stimulus_filenames = get_stimuli_filenames(stimuli)

        self.directory = directory
        files = [os.path.relpath(filename, start=directory) for filename in glob.glob(os.path.join(directory, '**', '*'), recursive=True)]
        stems = [os.path.splitext(f)[0] for f in files]

        stimuli_stems = [os.path.splitext(f)[0] for f in stimulus_filenames]
        stimuli_stems = get_keys_from_filenames(stimuli_stems, stems)

        if not set(stimuli_stems).issubset(stems):
            missing_predictions = set(stimuli_stems).difference(stems)
            raise ValueError("missing predictions for {}".format(missing_predictions))

        indices = [stems.index(f) for f in stimuli_stems]

        files = [os.path.join(directory, f) for f in files]
        files = [files[i] for i in indices]

        super(SaliencyMapModelFromDirectory, self).__init__(stimuli, files, **kwargs)


class SaliencyMapModelFromFile(SaliencyMapModel):
    """
    This class exposes a list of saliency maps stored in a .mat file
    as a pysaliency SaliencyMapModel. Especially, it can be used
    to import LSUN submissions into pysaliency.
    """
    def __init__(self, stimuli, filename, key='results', **kwargs):
        super(SaliencyMapModelFromFile, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.filename = filename
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.mat':
            self.load_matlab(filename, key=key)
        else:
            raise ValueError('Unkown filetype', filename)

    def load_matlab(self, filename, key='results'):
        import hdf5storage
        data = hdf5storage.loadmat(filename)[key]
        if len(data.shape) == 2 and len(self.stimuli) > 1:
            if data.shape[0] == 1:
                data = data[0]
            elif data.shape[1] == 1:
                data = data[:, 0]
            else:
                raise ValueError('Data has wrong shape: {} (need 1xN, Nx1 or N)'.format(data.shape))
        if len(data.shape) > 2:
            raise ValueError('Data has wrong shape: {} (need 1xN, Nx1 or N)'.format(data.shape))
        expected_shape = (len(self.stimuli),)
        if not data.shape == expected_shape:
            raise ValueError('Wrong number of saliency maps! Expected {}, got {}'.format(expected_shape, data.shape))
        self._saliency_maps = [data[i] for i in range(len(self.stimuli))]

    def _saliency_map(self, stimulus):
        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
        smap = self._saliency_maps[stimulus_index]
        if smap.shape != (stimulus.shape[0], stimulus.shape[1]):
            raise ValueError('Wrong shape!')
        return smap


class ModelFromDirectory(Model):
    def __init__(self, stimuli, directory, **kwargs):
        super(ModelFromDirectory, self).__init__(**kwargs)
        self.internal_model = SaliencyMapModelFromDirectory(stimuli, directory, caching=False)

    def _log_density(self, stimulus):
        smap = self.internal_model.saliency_map(stimulus)
        if not -0.01 <= logsumexp(smap) <= 0.01:
            raise ValueError('Not a correct log density!')
        return smap


def get_keys_recursive(group, prefix=''):
        import h5py

        keys = []

        for subgroup_name, subgroup in group.items():
            if isinstance(subgroup, h5py.Group):
                subprefix = f"{prefix}{subgroup_name}/"
                keys.extend(get_keys_recursive(subgroup, prefix=subprefix))
            else:
                keys.append(f"{prefix}{subgroup_name}")

        return keys


class HDF5SaliencyMapModel(SaliencyMapModel):
    """ exposes a HDF5 file with saliency maps as pysaliency model

        The stimuli have to be of type `FileStimuli`. For each
        stimulus file, the model expects a dataset with the same
        name in the dataset.
    """
    def __init__(self, stimuli, filename, check_shape=True, **kwargs):
        super(HDF5SaliencyMapModel, self).__init__(**kwargs)

        self.stimuli = stimuli
        self.filename = filename
        self.check_shape = check_shape

        if not os.path.isfile(self.filename):
            raise ValueError(f'File {self.filename} does not exist')

        import h5py
        self.hdf5_file = h5py.File(self.filename, 'r')
        self.all_keys = get_keys_recursive(self.hdf5_file)

        self.names = get_keys_from_filenames(get_stimuli_filenames(stimuli), self.all_keys)

    def _saliency_map(self, stimulus):
        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
        stimulus_key = self.names[stimulus_index]
        smap = self.hdf5_file[stimulus_key][:]
        if not smap.shape == (stimulus.shape[0], stimulus.shape[1]):
            if self.check_shape:
                warnings.warn('Wrong shape for stimulus {}'.format(stimulus_key), stacklevel=4)
        return smap


class HDF5Model(Model):
    """ exposes a HDF5 file with log densities as pysaliency model.

        For more detail see HDF5SaliencyMapModel
    """
    def __init__(self, stimuli, filename, check_shape=True, **kwargs):
        super(HDF5Model, self).__init__(**kwargs)
        self.parent_model = HDF5SaliencyMapModel(
            stimuli=stimuli,
            filename=filename,
            caching=False,
            check_shape=check_shape
        )

    def _log_density(self, stimulus):
        smap = self.parent_model.saliency_map(stimulus)
        if not -0.01 <= logsumexp(smap) <= 0.01:
            raise ValueError('Not a correct log density!')
        return smap


class TarFileLikeZipFile(object):
    """ Wrapper that makes TarFile behave more like ZipFile """
    def __init__(self, filename, *args, **kwargs):
        self.tarfile = tarfile.open(filename, *args, **kwargs)

    def namelist(self):
        filenames = []
        for tar_info in self.tarfile.getmembers():
            filenames.append(tar_info.name)
        return filenames

    def open(self, name, mode='r'):
        return self.tarfile.extractfile(name)


class PredictionsFromArchiveMixin(object):
    def __init__(self, stimuli, archive_file, *args, **kwargs):

        super(PredictionsFromArchiveMixin, self).__init__(*args, **kwargs)

        self.stimuli = stimuli
        self.stimulus_ids = list(stimuli.stimulus_ids)
        self.archive_file = archive_file
        _, archive_ext = os.path.splitext(self.archive_file)
        if archive_ext.lower() == '.zip':
            self.archive = zipfile.ZipFile(self.archive_file)
        elif archive_ext.lower() == '.tar':
            self.archive = TarFileLikeZipFile(self.archive_file)
        elif archive_ext.lower() == '.gz':
            self.archive = TarFileLikeZipFile(self.archive_file)
        elif archive_ext.lower() == '.rar':
            import rarfile
            self.archive = rarfile.RarFile(self.archive_file)
        else:
            raise ValueError(archive_file)

        files = self.archive.namelist()
        files = [f for f in files if '.ds_store' not in f.lower()]
        files = [f for f in files if '__macosx' not in f.lower()]
        stems = [os.path.splitext(f)[0] for f in files]

        stimuli_stems = [os.path.splitext(f)[0] for f in get_stimuli_filenames(stimuli)]
        stimuli_stems = get_keys_from_filenames_with_prefix(stimuli_stems, stems)

        prediction_filenames = []
        for stimuli_stem in stimuli_stems:
            candidates = [stem for stem in stems if stem.endswith(stimuli_stem)]
            if not candidates:
                raise ValueError("Can't find file for {}".format(stimuli_stem))
            if len(candidates) > 1:
                raise ValueError("Multiple candidates for {}: {}", stimuli_stem, candidates)

            target_stem, = candidates
            target_index = stems.index(target_stem)
            target_filename = files[target_index]

            prediction_filenames.append(target_filename)

        self.files = prediction_filenames

    def _prediction(self, stimulus):
        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
        filename = self.files[stimulus_index]
        return self._load_file(filename)

    def _load_file(self, filename):
        _, ext = os.path.splitext(filename)

        content = self.archive.open(filename)

        if ext.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
            return imread(content).astype(float)
        elif ext.lower() == '.npy':
            return np.load(content).astype(float)
        elif ext.lower() == '.mat':
            data = loadmat(content)
            variables = [v for v in data if not v.startswith('__')]
            if len(variables) > 1:
                raise ValueError('{} contains more than one variable: {}'.format(filename, variables))
            elif len(variables) == 0:
                raise ValueError('{} contains no data'.format(filename))
            return data[variables[0]]
        else:
            raise ValueError('Unkown file type: {}'.format(ext))

    @staticmethod
    def can_handle(filename):
        try:
            import rarfile
            return zipfile.is_zipfile(filename) or tarfile.is_tarfile(filename) or rarfile.is_rarfile(filename)
        except ImportError:
            warnings.warn("Can't test for rarfiles since rarfile is not installed!")
            return zipfile.is_zipfile(filename) or tarfile.is_tarfile(filename)


class SaliencyMapModelFromArchive(PredictionsFromArchiveMixin, SaliencyMapModel):
    def __init__(self, stimuli, archive_file, **kwargs):
        super(SaliencyMapModelFromArchive, self).__init__(stimuli, archive_file, **kwargs)

    def _saliency_map(self, stimulus):
        return self._prediction(stimulus)


class ModelFromArchive(PredictionsFromArchiveMixin, Model):
    def __init__(self, stimuli, archive_file, **kwargs):
        super(ModelFromArchive, self).__init__(stimuli, archive_file, **kwargs)

    def _log_density(self, stimulus):
        logdensity = self._prediction(stimulus)
        density_sum = logsumexp(logdensity)
        if np.abs(density_sum) > 0.01:
            logdensity_sum = np.sum(logdensity)
            raise ValueError("predictions not normalized: log of sum is {}, sum is {}".format(density_sum, logdensity_sum))
        return logdensity
