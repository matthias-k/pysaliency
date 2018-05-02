from __future__ import absolute_import, print_function, division

import unittest
import os.path
import shutil
import filecmp
from six.moves import cPickle
import six


def assert_equal(a, b):
    assert a == b


class TestWithData(unittest.TestCase):
    data_path = 'test_data'

    def setUp(self):
        if os.path.isdir(self.data_path):
            shutil.rmtree(self.data_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def tearDown(self):
        shutil.rmtree(self.data_path)

    def pickle_and_reload(self, data, pickler = cPickle):
        filename = os.path.join(self.data_path, 'object.pydat')

        with open(filename, 'wb') as f:
            pickler.dump(data, f)

        with open(filename, 'rb') as f:
            new_data = pickler.load(f)

        return new_data


def check_dircmp(dircmp):
    assert_equal(dircmp.left_only, [])
    assert_equal(dircmp.right_only, [])
    assert_equal(dircmp.diff_files, [])
    assert_equal(dircmp.funny_files, [])
    for sub_dcmp in dircmp.subdirs.values():
        check_dircmp(dircmp)


def assertDirsEqual(dir1, dir2, ignore=[]):
    if six.PY2:
        ignore = map(str, ignore)
    dircmp = filecmp.dircmp(dir1, dir2, ignore=ignore)
    check_dircmp(dircmp)
