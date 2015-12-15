from __future__ import print_function, division, absolute_import

import os.path

import numpy as np
from scipy.misc import imread
from scipy.io import loadmat

#from .models import Model
from .saliency_map_models import SaliencyMapModel
from .datasets import get_image_hash, FileStimuli


class SaliencyMapModelFromFiles(SaliencyMapModel):
    def __init__(self, stimuli, files):
        super(SaliencyMapModelFromFiles, self).__init__()
        self.stimuli = stimuli
        self.stimulus_ids = list(stimuli.stimulus_ids)
        self.files = files
        assert(len(files) == len(stimuli))

    def _saliency_map(self, stimulus):
        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
        filename = self.files[stimulus_index]
        return self._load_file(filename)

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
    def __init__(self, stimuli, directory):
        if not isinstance(stimuli, FileStimuli):
            raise TypeError('SaliencyMapModelFromDirectory works only with FileStimuli!')

        self.directory = directory
        files = os.listdir(directory)
        stems = [os.path.splitext(f)[0] for f in files]

        stimuli_files = [os.path.basename(f) for f in stimuli.filenames]
        stimuli_stems = [os.path.splitext(f)[0] for f in stimuli_files]

        assert set(stimuli_stems).issubset(stems)

        indices = [stems.index(f) for f in stimuli_stems]

        files = [os.path.join(directory, f) for f in files]
        files = [files[i] for i in indices]

        super(SaliencyMapModelFromDirectory, self).__init__(stimuli, files)
