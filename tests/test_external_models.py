from __future__ import absolute_import, print_function, division

import unittest

import numpy as np

import pysaliency
import pysaliency.external_datasets
from test_helpers import TestWithData


class TestExternalModels(TestWithData):
    def test_AIM_without_location(self):
        aim = pysaliency.AIM()
        stimulus = np.random.randn(600, 400, 3)
        saliency_map = aim.saliency_map(stimulus)
        self.assertEqual(saliency_map.shape, (600, 400))

    def test_AIM_with_location(self):
        aim = pysaliency.AIM(location=self.data_path)
        stimulus = np.random.randn(600, 400, 3)
        saliency_map = aim.saliency_map(stimulus)
        self.assertEqual(saliency_map.shape, (600, 400))

    def test_SUN_without_location(self):
        aim = pysaliency.SUN()
        stimulus = np.random.randn(600, 400, 3)
        saliency_map = aim.saliency_map(stimulus)
        self.assertEqual(saliency_map.shape, (600, 400))

    def test_SUN_with_location(self):
        aim = pysaliency.SUN(location=self.data_path)
        stimulus = np.random.randn(600, 400, 3)
        saliency_map = aim.saliency_map(stimulus)
        self.assertEqual(saliency_map.shape, (600, 400))

if __name__ == '__main__':
    unittest.main()
