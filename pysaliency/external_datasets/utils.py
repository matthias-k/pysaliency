from __future__ import absolute_import, print_function, division

import os
import shutil
import warnings

from ..datasets import FileStimuli, Stimuli, read_hdf5


def create_memory_stimuli(filenames, attributes=None):
    """
    Create a `Stimuli`-class from a list of filenames by reading the them
    """
    tmp_stimuli = FileStimuli(filenames)
    stimuli = list(tmp_stimuli.stimuli)  # Read all stimuli
    return Stimuli(stimuli, attributes=attributes)


def create_stimuli(stimuli_location, filenames, location=None, attributes=None):
    """
    Create a Stimuli class of stimuli.

    Parameters
    ----------

    @type  stimuli_location: string
    @param stimuli_location: the base path where the stimuli are located.
                             If `location` is provided, this directory will
                             be copied to `location` (see below).

    @type  filenames: list of strings
    @param filenames: lists the filenames of the stimuli to include in the dataset.
                      Filenames have to be relative to `stimuli_location`.

    @type  location: string or `None`
    @param location: If provided, the function will copy the filenames to
                     `location` and return a `FileStimuli`-object for the
                     copied files. Otherwise a `Stimuli`-object is returned.

    @returns: `Stimuli` or `FileStimuli` object depending on `location`.

    """
    if location is not None:
        shutil.copytree(stimuli_location,
                        location)
        filenames = [os.path.join(location, f) for f in filenames]

        return FileStimuli(filenames, attributes=attributes)

    else:
        filenames = [os.path.join(stimuli_location, f) for f in filenames]
        return create_memory_stimuli(filenames, attributes=attributes)


def _load(filename):
    """attempt to load hdf5 file and fallback to pickle files if present"""
    if os.path.isfile(filename):
        return read_hdf5(filename)

    stem, ext = os.path.splitext(filename)
    pydat_filename = stem + '.pydat'

    if os.path.isfile(pydat_filename):
        import dill
        # raise deprecation warning
        warnings.warn("Using pickle files is deprecated. Please convert to hdf5 files instead. Pickle support will be removed in pysaliency 0.4", DeprecationWarning, stacklevel=2)
        return dill.load(open(pydat_filename, 'rb'))
    else:
        raise FileNotFoundError(f"Neither {filename} nor {pydat_filename} exist.")