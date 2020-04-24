from __future__ import print_function, absolute_import, division
from collections import Sequence, MutableMapping
from itertools import chain
from glob import iglob
from contextlib import contextmanager, ExitStack
import warnings as _warnings
import os as _os
import sys as _sys
import os
import hashlib
from functools import partial
import warnings
import shutil
from six.moves import urllib, filterfalse, map
from six import iterkeys
import subprocess as sp
from tempfile import mkdtemp

import numpy as np

from boltons.cacheutils import LRU
import deprecation
from tqdm import tqdm
import requests


def build_padded_2d_array(arrays, max_length=None, padding_value=np.nan):
    if max_length is None:
        max_length = np.max([len(a) for a in arrays])

    output = np.ones((len(arrays), max_length), dtype=np.asarray(arrays[0]).dtype)
    output *= padding_value
    for i, array in enumerate(arrays):
        output[i, :len(array)] = array

    return output


def full_split(filename):
    """ split filename into all parts """
    path, basename = os.path.split(filename)
    components = [path, basename]
    while components[0] and components[0] != '/':
        first_part = components.pop(0)
        path, basename = os.path.split(first_part)
        components = [path, basename] + components

    if components[0] == '':
        components = components[1:]

    return components


def get_minimal_unique_filenames(filenames):
    if len(filenames) <= 1:
        return [os.path.basename(item) for item in filenames]

    components = [full_split(filename) for filename in filenames]

    while len(set(item[0] for item in components)) <= 1:
        components = [item[1:] for item in components]

    return [os.path.join(*item) for item in components]


def remove_trailing_nans(data):
    """Filters a scanpath arrays to remove the ending part of nans."""
    for i in range(len(data)):
        if np.all(np.isnan(data[i:])):
            return data[:i]
    return data


def lazy_property(fn):
    """Lazy property: Is only calculated when first used.
       Code from http://stackoverflow.com/questions/3012421/python-lazy-property-decorator"""
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


class LazyList(Sequence):
    """
    A list-like class that is able to generate it's entries only
    when needed. Entries can be cached.

    LayList implements `collections.Sequence` and therefore `__contains__`.
    However, to use it, in the worst case all elements of the list have
    to be generated.

    .. note::
        As `LazyList` stores the generator function, pickling it
        will usually fail. To pickle a `LazyList`, use `dill`.
    """
    def __init__(self, generator, length, cache=True, pickle_cache=False):
        """
        Parameters
        ----------

        @type  generator: callable
        @param generator: A function that takes an integer `n` and returns the
                          `n`-th element of the list.

        @type  length:   int
        @param length:   The length of the list

        @type  cache: bool, defaults to `True`
        @param cache: Wether to cache the list items.

        @type  pickle_cache: bool, defaults to `False`
        @param pickle_cache: Whether the cache should be saved when
                             pickling the object.
        """
        self.generator = generator
        self.length = length
        self.cache = cache
        self.pickle_cache = pickle_cache
        self._cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]
        elif isinstance(index, list):
            return [self[i] for i in index]
        else:
            return self._getitem(index)

    def _getitem(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        if index in self._cache:
            return self._cache[index]
        value = self.generator(index)
        if self.cache:
            self._cache[index] = value
        return value

    def __getstate__(self):
        # we don't want to save the cache
        state = dict(self.__dict__)
        if not self.pickle_cache:
            state.pop('_cache')
        return state

    def __setstate__(self, state):
        if not '_cache' in state:
            state['_cache'] = {}
        self.__dict__ = dict(state)


class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.

    From http://stackoverflow.com/a/19299884
    """

    def __init__(self, suffix="", prefix="tmp", dir=None, cleanup=True):
        self._closed = False
        self.name = None  # Handle mkdtemp raising an exception
        self.name = mkdtemp(suffix, prefix, dir)
        self.do_cleanup = cleanup

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        if not self.do_cleanup:
            return
        if self.name and not self._closed:
            try:
                self._rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                # Issue #10188: Emit a warning on stderr
                # if the directory could not be cleaned
                # up due to missing globals
                if "None" not in str(ex):
                    raise
                print("ERROR: {!r} while cleaning up {!r}".format(ex, self,),
                      file=_sys.stderr)
                return
            self._closed = True
            if _warn:
                self._warn("Implicitly cleaning up {!r}".format(self))

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

    # XXX (ncoghlan): The following code attempts to make
    # this class tolerant of the module nulling out process
    # that happens during CPython interpreter shutdown
    # Alas, it doesn't actually manage it. See issue #10188
    _listdir = staticmethod(_os.listdir)
    _path_join = staticmethod(_os.path.join)
    _isdir = staticmethod(_os.path.isdir)
    _islink = staticmethod(_os.path.islink)
    _remove = staticmethod(_os.remove)
    _rmdir = staticmethod(_os.rmdir)
    _warn = _warnings.warn

    def _rmtree(self, path):
        # Essentially a stripped down version of shutil.rmtree.  We can't
        # use globals because they may be None'ed out at shutdown.
        for name in self._listdir(path):
            fullname = self._path_join(path, name)
            try:
                isdir = self._isdir(fullname) and not self._islink(fullname)
            except OSError:
                isdir = False
            if isdir:
                self._rmtree(fullname)
            else:
                try:
                    self._remove(fullname)
                except OSError:
                    pass
        try:
            self._rmdir(path)
        except OSError:
            pass


@contextmanager
def atomic_directory_setup(directory):
    """ context manager that makes sure that directory is deleted in case of exceptions. """
    with ExitStack() as stack:
        if directory is not None:
            stack.callback(lambda: shutil.rmtree(directory))
        yield
        stack.pop_all()


def which(program):
    """
    Check whether a program is present on the system.
    from https://stackoverflow.com/a/377028
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def filter_files(filenames, ignores):
    """
    Filter a list of files, excluding all filenames which contain
    an element of `ignores` as part of their path
    """
    parts = map(full_split, filenames)
    inds = [i for i, ps in enumerate(parts)
            if not any([ignore in ps for ignore in ignores])]
    return [filenames[i] for i in inds]


class MatlabOptions(object):
    matlab_names = ['matlab', 'matlab.exe']
    octave_names = ['octave', 'octave.exe']


def get_matlab_or_octave():
    for name in MatlabOptions.matlab_names + MatlabOptions.octave_names:
        if which(name):
            return which(name)
    raise Exception('No version of matlab or octave was found on this system!')


def run_matlab_cmd(cmd, cwd=None):
    matlab = get_matlab_or_octave()
    args = []
    if os.path.basename(matlab).startswith('matlab'):
        args += ['-nodesktop', '-nosplash', '-r']
        args.append("try;{};catch exc;disp(getReport(exc));disp('__ERROR__');exit(1);end;quit".format(cmd))
    else:
        args += ['--traditional', '--eval']
        args.append("try;{};catch exc;struct_levels_to_print(10);print_struct_array_contents(true);disp(lasterror);for i=1:size(lasterror.stack);disp(lasterror.stack(i));end;disp('__ERROR__');exit(1);end;quit".format(cmd))
    sp.check_call([matlab] + args, cwd=cwd)


def check_file_hash(filename, md5_hash):
    """
    Check a file's hash and issue a warning it is has not the expected value.
    """
    print('Checking md5 sum...')
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        # read file in chunks to avoid "invalid argument" error
        # see https://stackoverflow.com/a/48123430
        for block in iter(partial(f.read, 64 * (1 << 20)), b''):
            hasher.update(block)

    file_hash = hasher.hexdigest()
    if file_hash != md5_hash:
        warnings.warn("MD5 sum of {} has changed. Expected {} but got {}. This might lead to"
                      " this code producing wrong data.".format(filename, md5_hash, file_hash))


def download_file_old(url, target):
    """Download url to target while displaying progress information."""
    class Log(object):
        def __init__(self):
            self.last_percent = -1

        def __call__(self, blocks_recieved, block_size, file_size):
            percent = int(blocks_recieved * block_size / file_size * 100)
            if percent == self.last_percent:
                return
            print('\rDownloading file. {}% done'.format(percent), end='')
            self.last_percent = percent
    urllib.request.urlretrieve(url, target, Log())
    print('')


def download_file(url, target):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    with open(target, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading file') as progress_bar:
            for data in r.iter_content(32*1024):
                f.write(data)
                progress_bar.update(32*1024)


def download_and_check(url, target, md5_hash):
    """Download url to target and check for correct md5_hash. Prints warning if hash is not correct."""
    download_file(url, target)
    check_file_hash(target, md5_hash)


def download_file_from_google_drive(id, destination):
    """adapted from https://drive.google.com/uc?id=0B2hsWbciDVedWHFiMUVVWFRZTE0&export=download"""
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with tqdm(unit='B', unit_scale=True) as pbar:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


class Cache(MutableMapping):
    """Cache that supports saving the items to files

    Set `cache_location` to save all newly set
    items to .npy files in cache_location.

    .. warning ::
        Items that have been set before setting `cache_location` won't
        be saved to files!

    """
    def __init__(self, cache_location=None, pickle_cache=False,
                 memory_cache_size=None):
        if memory_cache_size:
            self._cache = LRU(max_size=memory_cache_size)
        else:
            self._cache = {}
        self.cache_location = cache_location
        self.pickle_cache = pickle_cache

    def clear(self):
        """ Clear memory cache"""
        self._cache = {}

    def filename(self, key):
        return os.path.join(self.cache_location, '{}.npy'.format(key))

    def __getitem__(self, key):
        if not key in self._cache:
            if self.cache_location is not None:
                filename = self.filename(key)
                if os.path.exists(filename):
                    value = np.load(filename)
                    self._cache[key] = value
                else:
                    raise KeyError('Key {} neither in cache nor on disk'.format(key))
        return self._cache[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Only string keys are supported right now!')
        if self.cache_location is not None:
            if not os.path.exists(self.cache_location):
                os.makedirs(self.cache_location)
            filename = self.filename(key)
            np.save(filename, value)
        self._cache[key] = value

    def __delitem__(self, key):
        if self.cache_location is not None:
            filename = self.filename(key)
            if os.path.exists(filename):
                os.remove(filename)
        del self._cache[key]

    def __iter__(self):
        if self.cache_location is not None:
            filenames = iglob(self.filename('*'))
            keys = map(lambda f: os.path.splitext(os.path.basename(f))[0], filenames)
            new_keys = filterfalse(lambda key: key in self._cache.keys(), keys)
            return chain(iterkeys(self._cache), new_keys)
        else:
            return iterkeys(self._cache)

    def __len__(self):
        i = iter(self)
        return len(list(i))

    def __getstate__(self):
        # we don't want to save the cache
        state = dict(self.__dict__)
        if not self.pickle_cache:
            state.pop('_cache')
        return state

    def __setstate__(self, state):
        if not '_cache' in state:
            if state.get('memory_cache_size'):
                state['_cache'] = LRU(max_size=memory_cache_size)
            else:
                state['_cache'] = {}
        self.__dict__ = dict(state)


def average_values(values, fixations, average='fixation'):
    if average == 'fixation':
        return np.mean(values)
    elif average == 'image':
        import pandas as pd
        df = pd.DataFrame({'n': fixations.n, 'value': values})
        return df.groupby('n')['value'].mean().mean()
    else:
        raise ValueError(average)


def deprecated_class(deprecated_in=None, removed_in=None, current_version=None, details=''):
    def wrap(cls):
        class DeprecatedClass(cls):
            @deprecation.deprecated(deprecated_in=deprecated_in, removed_in=removed_in, current_version=current_version, details=details)
            def __init__(self, *args, **kwargs):
                super(DeprecatedClass, self).__init__(*args, **kwargs)

        return DeprecatedClass

    return wrap
