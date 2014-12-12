from __future__ import print_function, absolute_import, division
from collections import Sequence

import warnings as _warnings
import os as _os
import sys as _sys
import os
import hashlib
import warnings
from six.moves import urllib
import subprocess as sp
from tempfile import mkdtemp


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


def full_split(filename):
    """Split filename into all of its parts"""
    parts = list(os.path.split(filename))
    if parts[0]:
        return full_split(parts[0]) + [parts[1]]
    else:
        return [parts[1]]


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
        args.append("try;{};catch exc;disp(lasterror);for i=1:size(lasterror.stack);disp(lasterror.stack(i));end;disp('__ERROR__');exit(1);end;quit".format(cmd))
    sp.check_call([matlab] + args, cwd=cwd)


def check_file_hash(filename, md5_hash):
    """
    Check a file's hash and issue a warning it is has not the expected value.
    """
    print('Checking md5 sum...')
    with open(filename, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    if file_hash != md5_hash:
        warnings.warn("MD5 sum of {} has changed. Expected {} but got {}. This might lead to"
                      " this code producing wrong data.".format(filename, md5_hash, file_hash))


def download_file(url, target):
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


def download_and_check(url, target, md5_hash):
    """Download url to target and check for correct md5_hash. Prints warning if hash is not correct."""
    download_file(url, target)
    check_file_hash(target, md5_hash)
