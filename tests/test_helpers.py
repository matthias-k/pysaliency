from __future__ import absolute_import, print_function, division

import unittest
import os.path
import shutil
from six.moves import cPickle


class TestWithData(unittest.TestCase):
    data_path = 'test_data'

    def setUp(self):
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
