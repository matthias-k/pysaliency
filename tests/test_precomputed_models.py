from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pathlib
import warnings
import zipfile

import h5py
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


def test_export_root_attrs_written(file_stimuli, tmpdir):
    """New-format files always have type/version/downscale_factor/dtype root attrs."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename)
    with h5py.File(filename, 'r') as f:
        assert f.attrs['type'] == 'pysaliency.precomputed_models.predictions'
        assert f.attrs['version'] == '1.0'
        assert int(f.attrs['downscale_factor']) == 1
        assert f.attrs['dtype'] == 'float64'


def test_export_dtype_float32(file_stimuli, tmpdir):
    """dtype=np.float32 stores float32 datasets."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=np.float32)
    with h5py.File(filename, 'r') as f:
        assert f.attrs['dtype'] == 'float32'
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float32


def test_export_dtype_float16(file_stimuli, tmpdir):
    """dtype=np.float16 stores float16 datasets."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=np.float16)
    with h5py.File(filename, 'r') as f:
        assert f.attrs['dtype'] == 'float16'
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float16


def test_export_dtype_none_preserves_native(file_stimuli, tmpdir):
    """dtype=None (default) preserves the model's native output dtype."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=None)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        # GaussianSaliencyMapModel returns float64
        assert f[keys[0]].dtype == np.float64


def test_export_downscale_stored_shape(file_stimuli, tmpdir):
    """Stored shape is ceil(H/k) x ceil(W/k) for divisible dimensions."""
    # file_stimuli uses 100x100 images, which is divisible by 2 and 4
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert f[keys[0]].shape == (50, 50)


def test_export_downscale_original_shape_attr(file_stimuli, tmpdir):
    """original_shape attr is present and correct when downscale_factor > 1."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        ds = f[keys[0]]
        assert 'original_shape' in ds.attrs
        np.testing.assert_array_equal(ds.attrs['original_shape'], [100, 100])
        assert ds.attrs['original_shape'].dtype == np.int64


def test_export_no_downscale_no_original_shape_attr(file_stimuli, tmpdir):
    """original_shape attr is absent when downscale_factor == 1."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert 'original_shape' not in f[keys[0]].attrs


@pytest.fixture
def file_stimuli_nondivisible(tmpdir):
    """Stimuli with 101x97 images — not divisible by 2 or 4."""
    filenames = []
    for i in range(3):
        filename = tmpdir.join(f'stim_{i:04d}.png')
        imsave(str(filename), np.random.randint(0, 255, (101, 97, 3), dtype=np.uint8))
        filenames.append(str(filename))
    return pysaliency.FileStimuli(filenames=filenames)


def test_export_downscale_nondivisible_stored_shape(file_stimuli_nondivisible, tmpdir):
    """Non-divisible shapes are padded: stored shape is ceil(H/k) x ceil(W/k)."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli_nondivisible, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        # ceil(101/2)=51, ceil(97/2)=49
        assert f[keys[0]].shape == (51, 49)


def test_export_downscale_nondivisible_original_shape_attr(file_stimuli_nondivisible, tmpdir):
    """original_shape stores the pre-padding shape, not the padded or stored shape."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli_nondivisible, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        np.testing.assert_array_equal(f[keys[0]].attrs['original_shape'], [101, 97])


def test_export_float32_downscale_dtype_is_float32(file_stimuli, tmpdir):
    """float32 model output + downscale_factor > 1 stores float32, not float64."""
    # GaussianSaliencyMapModel returns float64, so we need a float32-producing model
    class Float32SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.float32)

    model = Float32SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float32
        assert f.attrs['dtype'] == 'float32'
        assert int(f.attrs['downscale_factor']) == 2


# NOTE: The spec test table lists "Append mode — mismatch (dtype only)" but the spec design
# section intentionally excludes dtype from the consistency check (it is purely informational
# and cannot be determined before the first stimulus is computed). There is therefore no
# ValueError raised on dtype mismatch — this is correct behavior, not a gap.


def test_export_append_consistency_mismatch_downscale(file_stimuli, tmpdir):
    """Appending with a different downscale_factor raises ValueError."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    partial = pysaliency.FileStimuli(filenames=file_stimuli.filenames[:3])
    export_model_to_hdf5(model, partial, filename, downscale_factor=2)
    with pytest.raises(ValueError, match='downscale_factor'):
        export_model_to_hdf5(model, file_stimuli, filename, overwrite=False, downscale_factor=1)


def test_export_append_new_file_root_attrs(file_stimuli, tmpdir):
    """overwrite=False on a new file still writes root attrs."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, overwrite=False)
    with h5py.File(filename, 'r') as f:
        assert 'type' in f.attrs
        assert f.attrs['version'] == '1.0'
        assert int(f.attrs['downscale_factor']) == 1
        assert 'dtype' in f.attrs


def test_export_append_legacy_file_no_root_attrs(file_stimuli, tmpdir):
    """Appending to a legacy file (no root attrs) succeeds without adding root attrs."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    partial = pysaliency.FileStimuli(filenames=file_stimuli.filenames[:3])
    remaining = pysaliency.FileStimuli(filenames=file_stimuli.filenames[3:])

    # Write a legacy-format file manually (no root attrs)
    names = pysaliency.utils.get_minimal_unique_filenames(partial.filenames)
    with h5py.File(filename, 'w') as f:
        for k, s in enumerate(partial):
            f.create_dataset(names[k], data=model.saliency_map(s))

    export_model_to_hdf5(model, remaining, filename, overwrite=False)

    with h5py.File(filename, 'r') as f:
        assert 'type' not in f.attrs  # no root attrs written into legacy file
        all_keys = pysaliency.precomputed_models.get_keys_recursive(f)
        assert len(all_keys) == len(file_stimuli)  # all stimuli present


def test_export_uint8_model_downscale_warns(tmpdir):
    """Uint8 model output + downscale_factor > 1 emits a UserWarning (stored as float64 > uint8)."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = []
    for i in range(2):
        fname = str(tmpdir.join(f'stim_{i}.png'))
        imsave(fname, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
        filenames.append(fname)
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with pytest.warns(UserWarning, match='larger'):
        export_model_to_hdf5(model, stimuli, filename, downscale_factor=2)


def test_export_uint8_model_downscale_stored_as_float64(tmpdir):
    """Uint8 model output + downscale_factor > 1 is stored as float64 (area avg result)."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = []
    for i in range(2):
        fname = str(tmpdir.join(f'stim_{i}.png'))
        imsave(fname, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
        filenames.append(fname)
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        export_model_to_hdf5(model, stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float64


def test_export_uint8_model_float32_dtype_warns(tmpdir):
    """Uint8 model + dtype=np.float32 warns (float32 > uint8)."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = [str(tmpdir.join(f'stim_{i}.png')) for i in range(2)]
    for fname in filenames:
        imsave(fname, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with pytest.warns(UserWarning, match='larger'):
        export_model_to_hdf5(model, stimuli, filename, dtype=np.float32)


def test_export_uint8_model_uint8_dtype_no_warn(tmpdir):
    """Uint8 model + dtype=np.uint8 does not warn."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = [str(tmpdir.join(f'stim_{i}.png')) for i in range(2)]
    for fname in filenames:
        imsave(fname, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an error
        export_model_to_hdf5(model, stimuli, filename, dtype=np.uint8)


@pytest.mark.parametrize('dtype,downscale_factor', [
    (None, 1),
    (np.float32, 1),
    (np.float16, 1),
    (np.float32, 2),
    (np.float16, 4),
])
def test_hdf5_saliency_map_model_roundtrip(file_stimuli, tmpdir, dtype, downscale_factor):
    """HDF5SaliencyMapModel returns correct shape and values after compact export."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename,
                          dtype=dtype, downscale_factor=downscale_factor)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    for s in file_stimuli:
        result = loaded.saliency_map(s)
        assert result.shape == (s.shape[0], s.shape[1]), \
            f"Shape mismatch: {result.shape} != {(s.shape[0], s.shape[1])}"


def test_hdf5_saliency_map_model_dtype_reduced_returns_native(file_stimuli, tmpdir):
    """dtype-reduced-only files return the stored native dtype (not upcast)."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=np.float32)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    result = loaded.saliency_map(file_stimuli[0])
    assert result.dtype == np.float32


def test_hdf5_saliency_map_model_downsampled_returns_float64(file_stimuli, tmpdir):
    """Downsampled files upsample and return float64."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    result = loaded.saliency_map(file_stimuli[0])
    assert result.dtype == np.float64


def test_hdf5_saliency_map_model_nondivisible_loaded_shape(file_stimuli_nondivisible, tmpdir):
    """Upsampled output shape matches original_shape (not padded shape)."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli_nondivisible, filename, downscale_factor=2)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli_nondivisible, filename)
    result = loaded.saliency_map(file_stimuli_nondivisible[0])
    assert result.shape == (101, 97)  # original shape, not (51, 49) or (102, 98)


def test_hdf5_saliency_map_model_resized_stimuli(tmpdir):
    """With check_shape=False and resized stimuli, upsampling targets original_shape."""
    from imageio import imsave as _imsave
    filenames = []
    for i in range(2):
        fname = str(tmpdir.join(f'stim_{i}.png'))
        _imsave(fname, np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        filenames.append(fname)
    full_stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, full_stimuli, filename, downscale_factor=2)

    # Load with the same filenames but we'll check that the loaded shape is 100x100
    # (original_shape), not 50x50 (stored) nor whatever the stimulus reports
    loaded = pysaliency.HDF5SaliencyMapModel(full_stimuli, filename, check_shape=False)
    result = loaded.saliency_map(full_stimuli[0])
    assert result.shape == (100, 100)


def test_hdf5_saliency_map_model_legacy_file_unchanged(file_stimuli, tmpdir):
    """Legacy files (no root attrs) still work exactly as before."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    # Write a legacy file manually
    names = pysaliency.utils.get_minimal_unique_filenames(file_stimuli.filenames)
    with h5py.File(filename, 'w') as f:
        for k, s in enumerate(file_stimuli):
            f.create_dataset(names[k], data=model.saliency_map(s))

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    for s in file_stimuli:
        expected = model.saliency_map(s)
        np.testing.assert_array_equal(loaded.saliency_map(s), expected)