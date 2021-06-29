from __future__ import division, print_function, absolute_import, unicode_literals

import os
import pathlib
import zipfile

import pytest

from imageio import imsave
import numpy as np

import pysaliency
from pysaliency import export_model_to_hdf5


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


def test_export_model_to_hdf5(file_stimuli, tmpdir):
    model = pysaliency.UniformModel()
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename)

    model2 = pysaliency.HDF5Model(file_stimuli, filename)
    for s in file_stimuli:
        np.testing.assert_allclose(model.log_density(s), model2.log_density(s))


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


def test_saliency_map_model_from_directory(file_stimuli, saliency_maps_in_directory):
    directory, predictions = saliency_maps_in_directory
    model = pysaliency.SaliencyMapModelFromDirectory(file_stimuli, directory)

    for stimulus_index, stimulus in enumerate(file_stimuli):
        expected = predictions[stimulus_index]
        actual = model.saliency_map(stimulus)
        np.testing.assert_equal(actual, expected)

@pytest.mark.skip("currently archivemodels can't handle same stimuli names in directory and subdirectory")
def test_saliency_map_model_from_archive(file_stimuli, saliency_maps_in_directory, tmpdir):
    directory, predictions = saliency_maps_in_directory

    archive = tmpdir / 'predictions.zip'

    # from https://stackoverflow.com/a/1855118
    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(path, '..')))
          
    with zipfile.ZipFile(str(archive), 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(str(directory), zipf)

    model = pysaliency.precomputed_models.SaliencyMapModelFromArchive(file_stimuli, str(archive))

    for stimulus_index, stimulus in enumerate(file_stimuli):
        expected = predictions[stimulus_index]
        actual = model.saliency_map(stimulus)
        np.testing.assert_equal(actual, expected)
