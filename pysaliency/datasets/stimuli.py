import json
import os
from collections.abc import Sequence
from hashlib import sha1
from typing import List, Union

import numpy as np

try:
    from imageio.v3 import imread
except ImportError:
    from imageio import imread
from PIL import Image
from tqdm import tqdm

from ..utils import LazyList
from .utils import create_hdf5_dataset, decode_string, hdf5_wrapper


def get_image_hash(img):
    """
    Calculate a unique hash for the given image.

    Can be used to cache results for images, e.g. saliency maps.
    """
    if isinstance(img, Stimulus):
        return img.stimulus_id
    return sha1(np.ascontiguousarray(img)).hexdigest()


class Stimulus(object):
    """
    Manages a stimulus.

    In application, this can always be substituted by
    the numpy array containing the actual stimulus. This class
    is just there to be able to cache calculation results and
    retrieve the cache content without having to load
    the actual stimulus
    """
    def __init__(self, stimulus_data, stimulus_id = None, shape = None, size = None):
        self.stimulus_data = stimulus_data
        self._stimulus_id = stimulus_id
        self._shape = shape
        self._size = size

    @property
    def stimulus_id(self):
        if self._stimulus_id is None:
            self._stimulus_id = get_image_hash(self.stimulus_data)
        return self._stimulus_id

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.stimulus_data.shape
        return self._shape

    @property
    def size(self):
        if self._size is None:
            self._size = self.stimulus_data.shape[0], self.stimulus_data.shape[1]
        return self._size


def as_stimulus(img_or_stimulus: Union[np.ndarray, Stimulus]) -> Stimulus:
    if isinstance(img_or_stimulus, Stimulus):
        return img_or_stimulus

    return Stimulus(img_or_stimulus)


class StimuliStimulus(Stimulus):
    """
    Stimulus bound to a Stimuli object
    """
    def __init__(self, stimuli, index):
        self.stimuli = stimuli
        self.index = index

    @property
    def stimulus_data(self):
        return self.stimuli.stimuli[self.index]

    @property
    def stimulus_id(self):
        return self.stimuli.stimulus_ids[self.index]

    @property
    def shape(self):
        return self.stimuli.shapes[self.index]

    @property
    def size(self):
        return self.stimuli.sizes[self.index]


class Stimuli(Sequence):
    """
    Manages a list of stimuli (i.e. images).

    The stimuli can be given as numpy arrays. Using the class `FileStimuli`, the stimuli
    can also be saved on disk and will be loaded only when needed.

    Attributes
    ----------
    stimuli :
        The stimuli as list of numpy arrays
    shapes :
        The shapes of the stimuli. For a grayscale stimulus this will
        be a 2-tuple, for a color stimulus a 3-tuple
    sizes :
        The sizes of all stimuli in pixels as pairs (height, width). In difference
        to `shapes`, the color dimension is ignored here.
    stimulus_ids:
        A unique id for each stimulus. Can be used to cache results for stimuli
    stimulus_objects:
        A `Stimulus` instance for each stimulus. Mainly for caching.

    """
    __attributes__ = []
    def __init__(self, stimuli, attributes=None):
        self.stimuli = stimuli
        self.shapes = [s.shape for s in self.stimuli]
        self.sizes = LazyList(lambda n: (self.shapes[n][0], self.shapes[n][1]),
                              length = len(self.stimuli))
        self.stimulus_ids = LazyList(lambda n: get_image_hash(self.stimuli[n]),
                                     length=len(self.stimuli),
                                     pickle_cache=True)
        self.stimulus_objects = [StimuliStimulus(self, n) for n in range(len(self.stimuli))]

        if attributes is not None:
            assert isinstance(attributes, dict)
            self.attributes = attributes
            self.__attributes__ = list(attributes.keys())
        else:
            self.attributes = {}

    def __len__(self):
        return len(self.stimuli)

    def _get_attribute_for_stimulus_subset(self, index):
        sub_attributes = {}
        for attribute_name, attribute_value in self.attributes.items():
            if isinstance(index, (list, np.ndarray)) and not isinstance(attribute_value, np.ndarray):
                sub_attributes[attribute_name] = [attribute_value[i] for i in index]
            else:
                sub_attributes[attribute_name] = attribute_value[index]

        return sub_attributes

    def __getitem__(self, index):
        if isinstance(index, slice):
            attributes = self._get_attribute_for_stimulus_subset(index)
            sub_stimuli = ObjectStimuli([self.stimulus_objects[i] for i in range(len(self))[index]], attributes=attributes)

            # populate stimulus_id cache with existing entries
            self._propagate_stimulus_ids(sub_stimuli, range(len(self))[index])

            return sub_stimuli
        elif isinstance(index, (list, np.ndarray)):
            index = np.asarray(index)
            if index.dtype == bool:
                if not len(index) == len(self.stimuli):
                    raise ValueError(f"Boolean index has to have the same length as the stimuli list but got {len(index)} and {len(self.stimuli)}")
                index = np.nonzero(index)[0]

            attributes = self._get_attribute_for_stimulus_subset(index)
            sub_stimuli = ObjectStimuli([self.stimulus_objects[i] for i in index], attributes=attributes)

            # populate stimulus_id cache with existing entries
            self._propagate_stimulus_ids(sub_stimuli, index)

            return sub_stimuli
        else:
            return self.stimulus_objects[index]

    def _propagate_stimulus_ids(self, sub_stimuli: "Stimuli", index: List[int]):
        for new_index, old_index in enumerate(index):
            if old_index in self.stimulus_ids._cache:
                sub_stimuli.stimulus_ids._cache[new_index] = self.stimulus_ids._cache[old_index]

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target, verbose=False, compression='gzip', compression_opts=9):
        """ Write stimuli to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.bytes_('Stimuli')
        target.attrs['version'] = np.bytes_('1.1')

        for n, stimulus in enumerate(tqdm(self.stimuli, disable=not verbose)):
            target.create_dataset(str(n), data=stimulus, compression=compression, compression_opts=compression_opts)

        self._attributes_to_hdf5(target)

        target.attrs['size'] = len(self)

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source):
        """ Read stimuli from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'Stimuli':
            raise ValueError("Invalid type! Expected 'Stimuli', got", data_type)

        if data_version not in ['1.0', '1.1']:
            raise ValueError("Invalid version! Expected '1.0' or '1.1', got", data_version)

        size = source.attrs['size']
        stimuli = []

        for n in range(size):
            stimuli.append(source[str(n)][...])

        __attributes__, attributes = cls._get_attributes_from_hdf5(source, data_version, '1.1')

        stimuli = cls(stimuli=stimuli, attributes=attributes)

        return stimuli

    def _attributes_to_hdf5(self, target):
        for attribute_name, attribute_value in self.attributes.items():
            create_hdf5_dataset(target, attribute_name, attribute_value)
        target.attrs['__attributes__'] = np.bytes_(json.dumps(self.__attributes__))

    @classmethod
    def _get_attributes_from_hdf5(cls, source, data_version, data_version_for_attribute_list):
        if data_version < data_version_for_attribute_list:
            __attributes__ = []
        else:
            json_attributes = source.attrs['__attributes__']
            if not isinstance(json_attributes, str):
                json_attributes = json_attributes.decode('utf8')
            __attributes__ = json.loads(json_attributes)

        attributes = {}
        for attribute in __attributes__:
            attribute_value = source[attribute][...]
            if isinstance(attribute_value.flatten()[0], bytes):
                attribute_shape = attribute_value.shape
                decoded_attribute_value = [decode_string(item) for item in attribute_value.flatten()]
                attribute_value = np.array(decoded_attribute_value).reshape(attribute_shape)
            attributes[attribute] = attribute_value

        return __attributes__, attributes


class ObjectStimuli(Stimuli):
    """
    This Stimuli class is mainly used for slicing of other stimuli objects.
    """
    def __init__(self, stimulus_objects, attributes=None):
        self.stimulus_objects = stimulus_objects
        self.stimuli = LazyList(lambda n: self.stimulus_objects[n].stimulus_data,
                                length = len(self.stimulus_objects))
        self.shapes = LazyList(lambda n: self.stimulus_objects[n].shape,
                               length = len(self.stimulus_objects))
        self.sizes = LazyList(lambda n: self.stimulus_objects[n].size,
                              length = len(self.stimulus_objects))
        self.stimulus_ids = LazyList(lambda n: self.stimulus_objects[n].stimulus_id,
                                     length = len(self.stimulus_objects))

        if attributes is not None:
            assert isinstance(attributes, dict)
            self.attributes = attributes
            self.__attributes__ = list(attributes.keys())
        else:
            self.attributes = {}


    def read_hdf5(self, target):
        raise NotImplementedError()


class FileStimuli(Stimuli):
    """
    Manage a list of stimuli that are saved as files.
    """
    def __init__(self, filenames, cached=True, shapes=None, attributes=None):
        """
        Create a stimuli object that reads it's stimuli from files.

        The stimuli are loaded lazy: each stimulus will be opened not
        before it is accessed. At creation time, all files are opened
        to read their dimensions, however the actual image data won't
        be read.

        .. note ::

            To calculate the stimulus_ids, the stimuli have to be
            loaded. Therefore it might be a good idea to load all
            stimuli and pickle the `FileStimuli` afterwarts. Then
            the ids are pickled but the stimuli will be reloaded
            when needed again.

        Parameters
        ----------
        filenames : list of strings
            filenames of the stimuli
        cache : bool, defaults to True
            whether loaded stimuli should be cached. The cache is excluded from pickling.
        """
        self.filenames = filenames
        self.stimuli = LazyList(self.load_stimulus, len(self.filenames), cache=cached)
        if shapes is None:
            self.shapes = []
            for f in filenames:
                img = Image.open(f)
                size = img.size
                if len(img.mode) > 1:
                    # PIL uses (width, height), we use (height, width)
                    self.shapes.append((size[1], size[0], len(img.mode)))
                else:
                    self.shapes.append((size[1], size[0]))
                del img
        else:
            self.shapes = shapes

        self.stimulus_ids = LazyList(lambda n: get_image_hash(self.stimuli[n]),
                                     length=len(self.stimuli),
                                     pickle_cache=True)
        self.stimulus_objects = [StimuliStimulus(self, n) for n in range(len(self.stimuli))]
        self.sizes = LazyList(lambda n: (self.shapes[n][0], self.shapes[n][1]),
                              length = len(self.stimuli))

        if attributes is not None:
            assert isinstance(attributes, dict)
            self.attributes = attributes
            self.__attributes__ = list(attributes.keys())
        else:
            self.attributes = {}

    @property
    def cached(self):
        return self.stimuli.cache

    @cached.setter
    def cached(self, value):
        self.stimuli.cache = value

    def load_stimulus(self, n):
        return imread(self.filenames[n])

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = list(range(len(self)))[index]

        if isinstance(index, (list, np.ndarray)):
            index = np.asarray(index)
            if index.dtype == bool:
                if not len(index) == len(self.stimuli):
                    raise ValueError(f"Boolean index has to have the same length as the stimuli list but got {len(index)} and {len(self.stimuli)}")
                index = np.nonzero(index)[0]

            filenames = [self.filenames[i] for i in index]
            shapes = [self.shapes[i] for i in index]
            attributes = self._get_attribute_for_stimulus_subset(index)
            sub_stimuli = type(self)(filenames=filenames, shapes=shapes, attributes=attributes, cached=self.cached)

            # populate stimulus_id cache with existing entries
            self._propagate_stimulus_ids(sub_stimuli, index)

            return sub_stimuli
        else:
            return self.stimulus_objects[index]

    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target):
        """ Write FileStimuli to hdf5 file or hdf5 group
        """

        target.attrs['type'] = np.bytes_('FileStimuli')
        target.attrs['version'] = np.bytes_('2.1')

        import h5py
        # make sure everything is unicode

        hdf5_filename = target.file.filename
        hdf5_directory = os.path.dirname(hdf5_filename)

        relative_filenames = [os.path.relpath(filename, hdf5_directory) for filename in self.filenames]
        decoded_filenames = [decode_string(filename) for filename in relative_filenames]
        encoded_filenames = [filename.encode('utf8') for filename in decoded_filenames]

        target.create_dataset(
            'filenames',
            data=np.array(encoded_filenames),
            dtype=h5py.special_dtype(vlen=str)
        )

        shape_dataset = target.create_dataset(
            'shapes',
            (len(self), ),
            dtype=h5py.special_dtype(vlen=np.dtype('int64'))
        )

        for n, shape in enumerate(self.shapes):
            shape_dataset[n] = np.array(shape)

        self._attributes_to_hdf5(target)

        target.attrs['size'] = len(self)

    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(cls, source, cached=True):
        """ Read FileStimuli from hdf5 file or hdf5 group """

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'FileStimuli':
            raise ValueError("Invalid type! Expected 'Stimuli', got", data_type)

        valid_versions = ['1.0', '2.0', '2.1']
        if data_version not in valid_versions:
            raise ValueError("Invalid version! Expected one of {}, got {}".format(', '.join(valid_versions), data_version))

        encoded_filenames = source['filenames'][...]

        filenames = [decode_string(filename) for filename in encoded_filenames]

        if data_version >= '2.0':
            hdf5_filename = source.file.filename
            hdf5_directory = os.path.dirname(hdf5_filename)
            filenames = [os.path.join(hdf5_directory, filename) for filename in filenames]

        shapes = [list(shape) for shape in source['shapes'][...]]

        __attributes__, attributes = cls._get_attributes_from_hdf5(source, data_version, '2.1')

        stimuli = cls(filenames=filenames, cached=cached, shapes=shapes, attributes=attributes)

        return stimuli


def check_prediction_shape(prediction: np.ndarray, stimulus: Union[np.ndarray, Stimulus]):
    stimulus = as_stimulus(stimulus)

    if prediction.shape != stimulus.size:
        raise ValueError(f"Prediction shape {prediction.shape} does not match stimulus shape {stimulus.size}")