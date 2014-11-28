from __future__ import absolute_import, print_function, division

import unittest
import dill

from pysaliency.utils import LazyList
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


if __name__ == '__main__':
    unittest.main()
