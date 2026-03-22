import pathlib
from weakref import WeakValueDictionary

from boltons.cacheutils import cached

from .datasets.fixations import Fixations, FixationTrains, ScanpathFixations
from .datasets.scanpaths import Scanpaths
from .datasets.stimuli import FileStimuli, Stimuli
from .datasets.utils import decode_string


@cached(WeakValueDictionary())
def _read_hdf5_from_file(source, **kwargs):
    import h5py
    with h5py.File(source, 'r') as hdf5_file:
        return read_hdf5(hdf5_file, **kwargs)


def _read_baseline_model(source, **kwargs):
    from .baseline_utils import BaselineModel
    return BaselineModel.read_hdf5(source, **kwargs)


def _read_crossvalidated_baseline_model(source, **kwargs):
    from .baseline_utils import CrossvalidatedBaselineModel
    return CrossvalidatedBaselineModel.read_hdf5(source, **kwargs)


_DATASET_READERS = {
    'Fixations': Fixations.read_hdf5,
    'ScanpathFixations': ScanpathFixations.read_hdf5,
    'FixationTrains': FixationTrains.read_hdf5,
    'Scanpaths': Scanpaths.read_hdf5,
    'Stimuli': Stimuli.read_hdf5,
    'FileStimuli': FileStimuli.read_hdf5,
}

_MODEL_READERS = {
    'pysaliency.baseline_utils.BaselineModel': _read_baseline_model,
    'pysaliency.baseline_utils.CrossvalidatedBaselineModel': _read_crossvalidated_baseline_model,
}


def read_hdf5(source, hdf5_kwargs=None, _expected_kind=None, **kwargs):
    if isinstance(source, (str, pathlib.Path)):
        try:
            return _read_hdf5_from_file(source, hdf5_kwargs=hdf5_kwargs, _expected_kind=_expected_kind, **kwargs)
        except TypeError:
            import h5py
            with h5py.File(source, 'r') as hdf5_file:
                return read_hdf5(hdf5_file, hdf5_kwargs=hdf5_kwargs, _expected_kind=_expected_kind, **kwargs)

    if hdf5_kwargs:
        raise NotImplementedError("Nested `hdf5_kwargs` routing is not implemented yet.")

    data_type = decode_string(source.attrs['type'])
    if data_type in _DATASET_READERS:
        if _expected_kind == 'model':
            raise ValueError("Invalid HDF model type:", data_type)
        return _DATASET_READERS[data_type](source, **kwargs)
    if data_type in _MODEL_READERS:
        if _expected_kind == 'dataset':
            raise ValueError("Invalid HDF dataset type:", data_type)
        return _MODEL_READERS[data_type](source, **kwargs)

    raise ValueError("Invalid HDF content type:", data_type)
