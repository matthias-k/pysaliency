from __future__ import absolute_import, print_function, division

import unittest
import os.path
import shutil
#from six.moves import cPickle
import dill as cPickle

from pysaliency.utils import LazyList


class TestLazyList(unittest.TestCase):
    data_path = 'test_data'
    def setUp(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def tearDown(self):
        shutil.rmtree(self.data_path)

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

    def test_pickle(self):
        def gen(i):
            print ('calling with {} yielding {}'.format(i, i**2))
            return i**2

        length = 20
        l = LazyList(gen, length)

        filename = os.path.join(self.data_path, 'lazy_list.pydat')
        with open(filename, 'wb') as f:
            cPickle.dump(l, f)

        with open(filename, 'rb') as f:
            l = cPickle.load(f)

        self.assertEqual(l._cache, {})
        self.assertEqual(list(l), [i**2 for i in range(length)])


if __name__ == '__main__':
    unittest.main()
