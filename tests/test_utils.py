from __future__ import absolute_import, print_function, division

import unittest
import dill
import glob
import os

import numpy as np

from pysaliency.utils import LazyList, TemporaryDirectory, Cache
from test_helpers import TestWithData


class TestLazyList(TestWithData):
    def test_lazy_list(self):
        calls = []

        def gen(i):
            calls.append(i)
            print ('calling with {} yielding {}'.format(i, i**2))
            return i**2

        length = 20

        l = LazyList(gen, length)
        self.assertEqual(len(l), length)

        for i in range(length):
            self.assertEqual(l[i], i**2)

        self.assertEqual(calls, range(length))

    def test_pickle_no_cache(self):
        def gen(i):
            print ('calling with {} yielding {}'.format(i, i**2))
            return i**2

        length = 20
        l = LazyList(gen, length)

        l = self.pickle_and_reload(l, pickler=dill)

        self.assertEqual(l._cache, {})
        self.assertEqual(list(l), [i**2 for i in range(length)])

    def test_pickle_with_cache(self):
        def gen(i):
            print ('calling with {} yielding {}'.format(i, i**2))
            return i**2

        length = 20
        l = LazyList(gen, length, pickle_cache=True)

        list(l)  # make sure all list items are generated

        l = self.pickle_and_reload(l, pickler=dill)

        self.assertEqual(l._cache, {i: i**2 for i in range(length)})
        self.assertEqual(list(l), [i**2 for i in range(length)])


class TestTemporaryDirectory(unittest.TestCase):
    def test_temporary_directory(self):
        with TemporaryDirectory() as tmp_dir:
            self.assertTrue(os.path.isdir(tmp_dir))

        self.assertFalse(os.path.isdir(tmp_dir))
        self.assertFalse(os.path.exists(tmp_dir))


class TestCache(TestWithData):
    def test_basics(self):
        cache = Cache()

        self.assertEqual(len(cache), 0)

        data = np.random.randn(10, 10, 3)
        cache['foo'] = data

        self.assertEqual(list(cache.keys()), ['foo'])
        np.testing.assert_allclose(cache['foo'], data)

        del cache['foo']

        self.assertEqual(len(cache), 0)

    def test_cache_to_disk(self):
        cache = Cache(cache_location = self.data_path)

        self.assertEqual(len(cache), 0)

        data = np.random.randn(10, 10, 3)
        cache['foo'] = data

        self.assertEqual(glob.glob(os.path.join(self.data_path, '*.*')),
                         [os.path.join(self.data_path, 'foo.npy')])

        self.assertEqual(list(cache.keys()), ['foo'])
        np.testing.assert_allclose(cache['foo'], data)

        cache = Cache(cache_location = self.data_path)
        self.assertEqual(cache._cache, {})
        self.assertEqual(list(cache.keys()), ['foo'])
        np.testing.assert_allclose(cache['foo'], data)

        del cache['foo']
        self.assertEqual(len(cache), 0)
        self.assertEqual(glob.glob(os.path.join(self.data_path, '*.*')),
                         [])

    def test_cache_to_disk_nonexisting_location(self):
        cache_location = os.path.join(self.data_path, 'cache')
        cache = Cache(cache_location = cache_location)

        self.assertEqual(len(cache), 0)

        data = np.random.randn(10, 10, 3)
        cache['foo'] = data

        self.assertEqual(glob.glob(os.path.join(cache_location, '*.*')),
                         [os.path.join(cache_location, 'foo.npy')])

        self.assertEqual(list(cache.keys()), ['foo'])
        np.testing.assert_allclose(cache['foo'], data)

        cache = Cache(cache_location = cache_location)
        self.assertEqual(cache._cache, {})
        self.assertEqual(list(cache.keys()), ['foo'])
        np.testing.assert_allclose(cache['foo'], data)

        del cache['foo']
        self.assertEqual(len(cache), 0)
        self.assertEqual(glob.glob(os.path.join(cache_location, '*.*')),
                         [])

    def test_pickle_cache(self):
        cache = Cache()

        self.assertEqual(len(cache), 0)

        data = np.random.randn(10, 10, 3)
        cache['foo'] = data

        self.assertEqual(list(cache.keys()), ['foo'])
        np.testing.assert_allclose(cache['foo'], data)

        cache2 = self.pickle_and_reload(cache)
        self.assertEqual(cache2._cache, {})
        self.assertEqual(len(cache2), 0)


    def test_pickle_cache_with_location(self):
        cache = Cache(cache_location = self.data_path)

        self.assertEqual(len(cache), 0)

        data = np.random.randn(10, 10, 3)
        cache['foo'] = data

        self.assertEqual(glob.glob(os.path.join(self.data_path, '*.*')),
                         [os.path.join(self.data_path, 'foo.npy')])

        self.assertEqual(list(cache.keys()), ['foo'])
        np.testing.assert_allclose(cache['foo'], data)

        cache2 = self.pickle_and_reload(cache)
        self.assertEqual(cache2._cache, {})
        self.assertEqual(len(cache2), 1)
        np.testing.assert_allclose(cache2['foo'], data)


if __name__ == '__main__':
    unittest.main()
