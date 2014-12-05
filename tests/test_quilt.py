from __future__ import print_function, division, unicode_literals, absolute_import

import os
import shutil
import filecmp

from pysaliency.quilt import PatchFile, QuiltSeries

from test_helpers import TestWithData


class TestPatchFile(TestWithData):
    def test_parsing(self):
        p = open('tests/test_quilt/patches/add_numbers.diff').read()
        patch = PatchFile(p)
        self.assertEqual(len(patch.diffs), 1)
        diff = patch.diffs[0]
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.source_filename, 'source.txt')
        self.assertEqual(diff.target_filename, 'source.txt')

        hunk = diff.hunks[0]
        self.assertEqual(hunk.source_start, 3)
        self.assertEqual(hunk.source_length, 6)

        self.assertEqual(hunk.target_start, 3)
        self.assertEqual(hunk.target_length, 8)

    def test_apply(self):
        location = os.path.join(self.data_path, 'patching')
        shutil.copytree('tests/test_quilt/source', location)
        p = open('tests/test_quilt/patches/add_numbers.diff').read()
        patch = PatchFile(p)

        patch.apply(location)
        self.assertTrue(filecmp.cmp(os.path.join(location, 'source.txt'),
                                    'tests/test_quilt/target/source.txt',
                                    shallow=False))


class TestSeries(TestWithData):
    def test_parsing(self):
        series = QuiltSeries(os.path.join('tests', 'test_quilt', 'patches'))
        self.assertEqual(len(series.patches), 1)

    def test_apply(self):
        location = os.path.join(self.data_path, 'patching')
        shutil.copytree('tests/test_quilt/source', location)
        series = QuiltSeries(os.path.join('tests', 'test_quilt', 'patches'))
        series.apply(location)
        to_compare = ['source.txt']
        match, mismatch, errors = filecmp.cmpfiles(location,
                                                   os.path.join('tests', 'test_quilt', 'target'),
                                                   to_compare,
                                                   shallow=False)
        self.assertEqual(mismatch, [])
        self.assertEqual(errors, [])
