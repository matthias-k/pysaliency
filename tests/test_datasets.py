from __future__ import absolute_import, print_function, division

import unittest
import os.path
from six.moves import cPickle
import dill

import numpy as np
from scipy.misc import imsave

import pysaliency
from test_helpers import TestWithData


class TestFixations(TestWithData):
    def test_from_fixations(self):
        xs_trains = [
            [0, 1, 2],
            [2, 2],
            [1, 5, 3]]
        ys_trains = [
            [10, 11, 12],
            [12, 12],
            [21, 25, 33]]
        ts_trains = [
            [0, 200, 600],
            [100, 400],
            [50, 500, 900]]
        ns = [0, 0, 1]
        subjects = [0, 1, 1]
        # Create Fixations
        f = pysaliency.Fixations.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)

        # Test fixation trains
        np.testing.assert_allclose(f.train_xs, [[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
        np.testing.assert_allclose(f.train_ys, [[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
        np.testing.assert_allclose(f.train_ts, [[0, 200, 600], [100, 400, np.nan], [50, 500, 900]])
        np.testing.assert_allclose(f.train_ns, [0, 0, 1])
        np.testing.assert_allclose(f.train_subjects, [0, 1, 1])

        # Test conditional fixations
        np.testing.assert_allclose(f.x, [0, 1, 2, 2, 2, 1, 5, 3])
        np.testing.assert_allclose(f.y, [10, 11, 12, 12, 12, 21, 25, 33])
        np.testing.assert_allclose(f.t, [0, 200, 600, 100, 400, 50, 500, 900])
        np.testing.assert_allclose(f.n, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(f.subjects, [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(f.lengths, [0, 1, 2, 0, 1, 0, 1, 2])
        np.testing.assert_allclose(f.x_hist, [[np.nan, np.nan],
                                              [0, np.nan],
                                              [0, 1],
                                              [np.nan, np.nan],
                                              [2, np.nan],
                                              [np.nan, np.nan],
                                              [1, np.nan],
                                              [1, 5]])

    def test_filter(self):
        def compare_fix(f1, f2, f2_inds):
            np.testing.assert_allclose(f1.x, f2.x[f2_inds])
            np.testing.assert_allclose(f1.y, f2.y[f2_inds])
            np.testing.assert_allclose(f1.t, f2.t[f2_inds])
            np.testing.assert_allclose(f1.n, f2.n[f2_inds])
            np.testing.assert_allclose(f1.subjects, f2.subjects[f2_inds])
        xs_trains = []
        ys_trains = []
        ts_trains = []
        ns = []
        subjects = []
        for n in range(1000):
            size = np.random.randint(10)
            xs_trains.append(np.random.randn(size))
            ys_trains.append(np.random.randn(size))
            ts_trains.append(np.cumsum(np.square(np.random.randn(size))))
            ns.append(np.random.randint(20))
            subjects.append(np.random.randint(20))
        f = pysaliency.Fixations.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)
        # First order filtering
        inds = f.n == 10
        _f = f.filter(inds)
        compare_fix(_f, f, inds)

        # second order filtering
        inds = np.nonzero(f.n == 10)[0]
        _f = f.filter(inds)
        inds2 = np.nonzero(_f.subjects == 0)[0]
        __f = _f.filter(inds2)
        cum_inds = inds[inds2]
        compare_fix(__f, f, cum_inds)

    def test_save_and_load(self):
        xs_trains = [
            [0, 1, 2],
            [2, 2],
            [1, 5, 3]]
        ys_trains = [
            [10, 11, 12],
            [12, 12],
            [21, 25, 33]]
        ts_trains = [
            [0, 200, 600],
            [100, 400],
            [50, 500, 900]]
        ns = [0, 0, 1]
        subjects = [0, 1, 1]
        # Create Fixations
        f = pysaliency.Fixations.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)

        filename = os.path.join(self.data_path, 'fixation.pydat')
        with open(filename, 'wb') as out_file:
            cPickle.dump(f, out_file)

        with open(filename, 'rb') as in_file:
            f = cPickle.load(in_file)
        # Test fixation trains
        np.testing.assert_allclose(f.train_xs, [[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
        np.testing.assert_allclose(f.train_ys, [[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
        np.testing.assert_allclose(f.train_ts, [[0, 200, 600], [100, 400, np.nan], [50, 500, 900]])
        np.testing.assert_allclose(f.train_ns, [0, 0, 1])
        np.testing.assert_allclose(f.train_subjects, [0, 1, 1])

        # Test conditional fixations
        np.testing.assert_allclose(f.x, [0, 1, 2, 2, 2, 1, 5, 3])
        np.testing.assert_allclose(f.y, [10, 11, 12, 12, 12, 21, 25, 33])
        np.testing.assert_allclose(f.t, [0, 200, 600, 100, 400, 50, 500, 900])
        np.testing.assert_allclose(f.n, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(f.subjects, [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(f.lengths, [0, 1, 2, 0, 1, 0, 1, 2])
        np.testing.assert_allclose(f.x_hist, [[np.nan, np.nan],
                                              [0, np.nan],
                                              [0, 1],
                                              [np.nan, np.nan],
                                              [2, np.nan],
                                              [np.nan, np.nan],
                                              [1, np.nan],
                                              [1, 5]])


class TestStimuli(TestWithData):
    def test_stimuli(self):
        img1 = np.random.randn(100, 200, 3)
        img2 = np.random.randn(50, 150)
        stimuli = pysaliency.Stimuli([img1, img2])

        self.assertEqual(stimuli.stimuli, [img1, img2])
        self.assertEqual(stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(stimuli.sizes, [(100, 200), (50, 150)])

        new_stimuli = self.pickle_and_reload(stimuli)
        print(new_stimuli.stimuli)

        self.assertEqual(len(new_stimuli.stimuli), 2)
        for s1, s2 in zip(new_stimuli.stimuli, [img1, img2]):
            np.testing.assert_allclose(s1, s2)
        self.assertEqual(new_stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(new_stimuli.sizes, [(100, 200), (50, 150)])


class TestFileStimuli(TestWithData):
    def test_file_stimuli(self):
        img1 = np.random.randint(255, size=(100, 200, 3)).astype('uint8')
        filename1 = os.path.join(self.data_path, 'img1.png')
        imsave(filename1, img1)

        img2 = np.random.randint(255, size=(50, 150)).astype('uint8')
        filename2 = os.path.join(self.data_path, 'img2.png')
        imsave(filename2, img2)

        stimuli = pysaliency.FileStimuli([filename1, filename2])

        self.assertEqual(len(stimuli.stimuli), 2)
        for s1, s2 in zip(stimuli.stimuli, [img1, img2]):
            np.testing.assert_allclose(s1, s2)
        self.assertEqual(stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(stimuli.sizes, [(100, 200), (50, 150)])

        new_stimuli = self.pickle_and_reload(stimuli, pickler=dill)
        print(new_stimuli.stimuli)

        self.assertEqual(len(new_stimuli.stimuli), 2)
        for s1, s2 in zip(new_stimuli.stimuli, [img1, img2]):
            np.testing.assert_allclose(s1, s2)
        self.assertEqual(new_stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(new_stimuli.sizes, [(100, 200), (50, 150)])


if __name__ == '__main__':
    unittest.main()
