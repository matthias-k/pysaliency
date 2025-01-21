from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pathlib
import zipfile

import numpy as np
import pytest
from imageio import imsave

import pysaliency
from pysaliency import export_model_to_hdf5


class TestSaliencyMapModel(pysaliency.SaliencyMapModel):
    def _saliency_map(self, stimulus):
        stimulus_data = pysaliency.datasets.as_stimulus(stimulus).stimulus_data
        if stimulus_data.ndim == 3:
            return stimulus_data.mean(axis=-1).astype(float)
        else:
            return np.array(stimulus_data, dtype=float)


@pytest.fixture
def file_stimuli(tmpdir):
    filenames = []
    for i in range(3):
        filename = tmpdir.join('stimulus_{:04d}.png'.format(i))
        imsave(str(filename), np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
        filenames.append(str(filename))

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('stimulus_{:04d}.png'.format(i))
            imsave(str(filename), np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))
    return pysaliency.FileStimuli(filenames=filenames)


@pytest.fixture
def stimuli_with_filenames(tmpdir):
    filenames = []
    stimuli = []
    for i in range(3):
        filename = tmpdir.join('stimulus_{:04d}.png'.format(i))
        stimuli.append(np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
        filenames.append(str(filename))

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_directory_{:04d}'.format(sub_directory_index))
        for i in range(5):
            filename = sub_directory.join('stimulus_{:04d}.png'.format(i))
            stimuli.append(np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))
    return pysaliency.Stimuli(stimuli=stimuli, attributes={'filenames': filenames})


@pytest.fixture(params=['FileStimuli', 'attributes'])
def stimuli(file_stimuli, stimuli_with_filenames, request):
    if request.param == 'FileStimuli':
        return file_stimuli
    elif request.param == 'attributes':
        return stimuli_with_filenames
    else:
        raise ValueError(request.param)


@pytest.fixture
def sub_stimuli(stimuli):
    unique_filenames = pysaliency.utils.get_minimal_unique_filenames(
        pysaliency.precomputed_models.get_stimuli_filenames(stimuli)
    )
    return stimuli[[i for i, f in enumerate(unique_filenames) if f.startswith('sub_directory_0001')]]


@pytest.fixture
def saliency_maps_in_directory(file_stimuli, tmpdir):
    stimuli_files = pysaliency.utils.get_minimal_unique_filenames(file_stimuli.filenames)

    prediction_dir = tmpdir.join('predictions')
    prediction_dir.mkdir()
    predictions = []
    rst = np.random.RandomState(seed=42)
    for filename in stimuli_files:
        prediction = rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8)
        target_name = prediction_dir.join(filename)
        pathlib.Path(target_name).resolve().parent.mkdir(exist_ok=True)
        imsave(str(target_name), prediction)
        predictions.append(prediction)

    return prediction_dir, predictions


def test_export_model_to_hdf5(stimuli, tmpdir):
    model = pysaliency.models.SaliencyMapNormalizingModel(TestSaliencyMapModel())
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, stimuli, filename)

    model2 = pysaliency.HDF5Model(stimuli, filename)
    for s in stimuli:
        np.testing.assert_allclose(model.log_density(s), model2.log_density(s))


def test_hdf5_model_sub_stimuli(stimuli, sub_stimuli, tmpdir):
    model = pysaliency.models.SaliencyMapNormalizingModel(TestSaliencyMapModel())
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, stimuli, filename)

    model2 = pysaliency.HDF5Model(sub_stimuli, filename)
    for s in sub_stimuli:
        np.testing.assert_allclose(model.log_density(s), model2.log_density(s))


def test_hdf5_model_empty_stimuli(stimuli, tmpdir):
    model = pysaliency.models.SaliencyMapNormalizingModel(TestSaliencyMapModel())
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, stimuli, filename)

    sub_stimuli = stimuli[[]]

    pysaliency.HDF5Model(sub_stimuli, filename)


def test_export_model_overwrite(file_stimuli, tmpdir):
    model1 = pysaliency.GaussianSaliencyMapModel(width=0.1)
    model2 = pysaliency.GaussianSaliencyMapModel(width=0.8)

    filename = str(tmpdir.join('model.hdf5'))

    partial_stimuli = pysaliency.FileStimuli(filenames=file_stimuli.filenames[:10])

    export_model_to_hdf5(model1, partial_stimuli, filename)
    export_model_to_hdf5(model2, file_stimuli, filename)

    model3 = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    for s in file_stimuli:
        np.testing.assert_allclose(model2.saliency_map(s), model3.saliency_map(s))


def test_export_model_no_overwrite(file_stimuli, tmpdir):
    model1 = pysaliency.GaussianSaliencyMapModel(width=0.1)
    model2 = pysaliency.GaussianSaliencyMapModel(width=0.8)

    filename = str(tmpdir.join('model.hdf5'))

    partial_stimuli = pysaliency.FileStimuli(filenames=file_stimuli.filenames[:5])

    export_model_to_hdf5(model1, partial_stimuli, filename)
    export_model_to_hdf5(model2, file_stimuli, filename, overwrite=False)

    model3 = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    for k, s in enumerate(file_stimuli):
        if k < 5:
            np.testing.assert_allclose(model1.saliency_map(s), model3.saliency_map(s))
        else:
            np.testing.assert_allclose(model2.saliency_map(s), model3.saliency_map(s))


def test_hdf5_model_sub_stimuli_different_prefix(tmpdir):
    rst = np.random.RandomState(seed=42)
    filenames = []
    for i in range(3):
        filename = tmpdir.join('stimulus_{:04d}.png'.format(i))
        imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
        filenames.append(str(filename))

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('stimulus_{}_{:04d}.png'.format(sub_directory_index, i))
            imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    rst = np.random.RandomState(seed=42)
    filenames = []
    for i in range(3):
        filename = tmpdir.join('stimulus_{:04d}.png'.format(i))
        imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
        filenames.append(str(filename))

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_prefix_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('stimulus_{}_{:04d}.png'.format(sub_directory_index, i))
            imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))

    stimuli_different_prefix = pysaliency.FileStimuli(filenames=filenames)
    sub_stimuli = stimuli_different_prefix[[i for i, f in enumerate(stimuli_different_prefix.filenames) if 'sub_prefix_directory_0001' in f]]

    model = pysaliency.models.SaliencyMapNormalizingModel(TestSaliencyMapModel())
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, stimuli, filename)

    model2 = pysaliency.HDF5Model(sub_stimuli, filename)
    for s in sub_stimuli:
        np.testing.assert_allclose(model.log_density(s), model2.log_density(s))


def test_hdf5_model_wrong_keys(tmpdir):
    rst = np.random.RandomState(seed=42)
    filenames = []

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('stimulus_{}_{:04d}.png'.format(sub_directory_index, i))
            imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    rst = np.random.RandomState(seed=42)
    filenames = []

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_prefix_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('other_stimulus_{}_{:04d}.png'.format(sub_directory_index, i))
            imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))

    stimuli_different_names = pysaliency.FileStimuli(filenames=filenames)

    model = pysaliency.models.SaliencyMapNormalizingModel(TestSaliencyMapModel())
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, stimuli, filename)

    with pytest.raises(pysaliency.precomputed_models.NoCommonPrefixError):
        pysaliency.HDF5Model(stimuli_different_names, filename)


def test_hdf5_model_sub_stimuli_different_prefix_nonunique(tmpdir):
    rst = np.random.RandomState(seed=42)
    filenames = []

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('stimulus_{:04d}.png'.format(i))
            imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    rst = np.random.RandomState(seed=42)
    filenames = []

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_prefix_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('stimulus_{:04d}.png'.format(i))
            imsave(str(filename), rst.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))

    stimuli_different_prefix = pysaliency.FileStimuli(filenames=filenames)
    sub_stimuli = stimuli_different_prefix[[i for i, f in enumerate(stimuli_different_prefix.filenames) if 'sub_prefix_directory_0001' in f]]

    model = pysaliency.models.SaliencyMapNormalizingModel(TestSaliencyMapModel())
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, stimuli, filename)

    with pytest.raises(pysaliency.precomputed_models.NonUniqueKeysError):
        pysaliency.HDF5Model(sub_stimuli, filename)


def test_saliency_map_model_from_directory(stimuli, saliency_maps_in_directory):
    directory, predictions = saliency_maps_in_directory
    model = pysaliency.SaliencyMapModelFromDirectory(stimuli, directory)

    for stimulus_index, stimulus in enumerate(stimuli):
        expected = predictions[stimulus_index]
        actual = model.saliency_map(stimulus)
        np.testing.assert_equal(actual, expected)


def test_saliency_map_model_from_directory_sub_stimuli(stimuli, sub_stimuli, saliency_maps_in_directory):
    directory, predictions = saliency_maps_in_directory
    full_model = pysaliency.SaliencyMapModelFromDirectory(stimuli, directory)
    sub_model = pysaliency.SaliencyMapModelFromDirectory(sub_stimuli, directory)

    for stimulus in sub_stimuli:
        expected = full_model.saliency_map(stimulus)
        actual = sub_model.saliency_map(stimulus)
        np.testing.assert_equal(actual, expected)


def test_saliency_map_model_from_archive(stimuli, saliency_maps_in_directory, tmpdir):
    directory, predictions = saliency_maps_in_directory

    archive = tmpdir / 'predictions.zip'

    # from https://stackoverflow.com/a/1855118
    def zipdir(path, ziph):
        for root, _, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))

    with zipfile.ZipFile(str(archive), 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(str(directory), zipf)

    model = pysaliency.precomputed_models.SaliencyMapModelFromArchive(stimuli, str(archive))

    for stimulus_index, stimulus in enumerate(stimuli):
        expected = predictions[stimulus_index]
        actual = model.saliency_map(stimulus)
        np.testing.assert_equal(actual, expected)


def test_saliency_map_model_from_archive_sub_stimuli(stimuli, sub_stimuli, saliency_maps_in_directory, tmpdir):
    directory, predictions = saliency_maps_in_directory

    archive = tmpdir / 'predictions.zip'

    # from https://stackoverflow.com/a/1855118
    def zipdir(path, ziph):
        for root, _, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))

    with zipfile.ZipFile(str(archive), 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(str(directory), zipf)

    full_model = pysaliency.precomputed_models.SaliencyMapModelFromArchive(stimuli, str(archive))
    sub_model = pysaliency.precomputed_models.SaliencyMapModelFromArchive(sub_stimuli, str(archive))

    for stimulus in sub_stimuli:
        expected = full_model.saliency_map(stimulus)
        actual = sub_model.saliency_map(stimulus)
        np.testing.assert_equal(actual, expected)