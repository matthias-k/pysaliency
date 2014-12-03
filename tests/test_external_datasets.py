from __future__ import absolute_import, print_function, division

import unittest
import os.path

import pysaliency
import pysaliency.external_datasets
from test_helpers import TestWithData


class TestExternalDatasets(TestWithData):
    def test_toronto_without_location(self):
        stimuli, fixations = pysaliency.external_datasets.get_toronto(location=None)
        self.assertIsInstance(stimuli, pysaliency.Stimuli)
        self.assertNotIsInstance(stimuli, pysaliency.FileStimuli)

        self.assertEqual(len(stimuli.stimuli), 120)
        for n in range(len(stimuli.stimuli)):
            self.assertEqual(stimuli.shapes[n], (511, 681, 3))
            self.assertEqual(stimuli.sizes[n], (511, 681))

    def test_toronto_with_location(self):
        stimuli, fixations = pysaliency.external_datasets.get_toronto(location=self.data_path)
        self.assertIsInstance(stimuli, pysaliency.Stimuli)
        self.assertIsInstance(stimuli, pysaliency.FileStimuli)
        self.assertTrue(os.path.isdir(os.path.join(self.data_path, 'toronto')))

        self.assertEqual(len(stimuli.stimuli), 120)
        for n in range(len(stimuli.stimuli)):
            self.assertEqual(stimuli.shapes[n], (511, 681, 3))
            self.assertEqual(stimuli.sizes[n], (511, 681))

    def test_mit_without_location(self):
        stimuli, fixations = pysaliency.external_datasets.get_mit1003(location=None)
        self.assertIsInstance(stimuli, pysaliency.Stimuli)
        self.assertNotIsInstance(stimuli, pysaliency.FileStimuli)

        self.assertEqual(len(stimuli.stimuli), 1003)
        for n in range(len(stimuli.stimuli)):
            self.assertEqual(max(stimuli.sizes[n]), 1024)

        self.assertEqual(len(fixations.x), 104171)
        self.assertEqual((fixations.n == 0).sum(), 121)

    def test_mit_with_location(self):
        stimuli, fixations = pysaliency.external_datasets.get_mit1003(location=self.data_path)
        self.assertIsInstance(stimuli, pysaliency.Stimuli)
        self.assertIsInstance(stimuli, pysaliency.FileStimuli)
        self.assertTrue(os.path.isdir(os.path.join(self.data_path, 'MIT1003')))

        self.assertEqual(len(stimuli.stimuli), 1003)
        for n in range(len(stimuli.stimuli)):
            self.assertEqual(max(stimuli.sizes[n]), 1024)

        self.assertEqual(len(fixations.x), 104171)
        self.assertEqual((fixations.n == 0).sum(), 121)

    def test_mit_onesize_without_location(self):
        stimuli, fixations = pysaliency.external_datasets.get_mit1003_onesize(location=None)
        self.assertIsInstance(stimuli, pysaliency.Stimuli)
        self.assertNotIsInstance(stimuli, pysaliency.FileStimuli)

        self.assertEqual(len(stimuli.stimuli), 463)
        for n in range(len(stimuli.stimuli)):
            self.assertEqual(stimuli.sizes[n], (768, 1024))

        self.assertEqual(len(fixations.x), 48771)
        self.assertEqual((fixations.n == 0).sum(), 121)

    def test_mit_onesize_with_location(self):
        stimuli, fixations = pysaliency.external_datasets.get_mit1003_onesize(location=self.data_path)
        self.assertIsInstance(stimuli, pysaliency.Stimuli)
        self.assertIsInstance(stimuli, pysaliency.FileStimuli)
        self.assertTrue(os.path.isdir(os.path.join(self.data_path, 'MIT1003_onesize')))

        self.assertEqual(len(stimuli.stimuli), 463)
        for n in range(len(stimuli.stimuli)):
            self.assertEqual(stimuli.sizes[n], (768, 1024))

        self.assertEqual(len(fixations.x), 48771)
        self.assertEqual((fixations.n == 0).sum(), 121)

if __name__ == '__main__':
    unittest.main()
