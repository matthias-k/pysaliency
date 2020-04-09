from __future__ import division, print_function, absolute_import, unicode_literals

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
