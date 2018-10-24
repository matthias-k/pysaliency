from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

from pysaliency.baseline_utils import fill_fixation_map


def test_fixation_map():
    fixations = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 2],
        [1, 2],
        [2, 1]])

    fixation_map = np.zeros((3, 3))
    fill_fixation_map(fixation_map, fixations)

    np.testing.assert_allclose(fixation_map, np.array([
        [1, 0, 0],
        [0, 2, 2],
        [0, 1, 0]]))
