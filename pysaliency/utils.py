from __future__ import print_function, absolute_import, division
from collections import Sequence

import warnings as _warnings
import os as _os
import sys as _sys
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

    def __init__(self, suffix="", prefix="tmp", dir=None):
        self._closed = False
        self.name = None  # Handle mkdtemp raising an exception
        self.name = mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
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
