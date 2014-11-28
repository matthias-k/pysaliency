from __future__ import print_function, absolute_import, division
from collections import Sequence


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
