"""
Code to apply quilt patches to files

This module enables pysaliency to use quilt patches
to patch code from external saliency models. While
in Linux, quilt itself could be used to apply the patches,
in Windows and Mac quilt might not be available and
nontrivial to install for users.

It does not support all possible patch files but only
the subset of functionality needed by pysaliency.
"""

from __future__ import absolute_import, print_function, division, unicode_literals

import os.path

from .utils import full_split


class Hunk(object):
    def __init__(self, lines):
        meta_data = lines.pop(0)
        a, src_data, target_data, b = meta_data.split()
        assert a == '@@'
        assert b == '@@'
        start, length = self.parse_position(src_data)
        assert start < 0
        self.source_start = -start
        self.source_length = length

        start, length = self.parse_position(target_data)
        assert start > 0
        self.target_start = start
        self.target_length = length

        self.lines = lines

    def parse_position(self, position):
        start, length = position.split(',')
        start = int(start)
        length = int(length)
        return start, length

    def apply(self, source, target):
        src_pos = self.source_start - 1
        assert len(target) == self.target_start - 1
        for line in self.lines:
            type, data = line[0], line[1:]
            if type == ' ':
                assert source[src_pos] == data
                target.append(data)
                src_pos += 1
            elif type == '-':
                assert source[src_pos] == data
                src_pos += 1
            elif type == '+':
                target.append(data)
            elif type == '\\':
                # Newline stuff, ignore
                pass
            else:
                raise ValueError(line)
        assert src_pos == self.source_start + self.source_length - 1
        assert len(target) == self.target_start + self.target_length - 1
        return src_pos


class Diff(object):
    def __init__(self, lines):
        source = lines.pop(0)
        assert source.startswith('--- ')
        _, source = source.split('--- ', 1)
        source, _ = source.split('\t', 1)
        source = os.path.join(*full_split(source)[1:])
        target = lines.pop(0)
        assert target.startswith('+++ ')
        _, target = target.split('+++ ', 1)
        target, _ = target.split('\t', 1)
        target = os.path.join(*full_split(target)[1:])
        self.source_filename = source
        self.target_filename = target
        self.hunks = []
        while lines:
            assert lines[0].startswith('@@ ')
            hunk_lines = [lines.pop(0)]
            while lines and not lines[0].startswith('@@ '):
                line = lines.pop(0)
                if line:
                    hunk_lines.append(line)
            self.hunks.append(Hunk(hunk_lines))

    def apply(self, location):
        hunks = list(self.hunks)
        source = open(os.path.join(location, self.source_filename)).read()
        source = source.split('\n')
        target = []
        src_pos = 0
        while src_pos < len(source):
            if hunks:
                if hunks[0].source_start == src_pos+1:
                    hunk = hunks.pop(0)
                    src_pos = hunk.apply(source, target)
                    continue
            target.append(source[src_pos])
            src_pos += 1
        open(os.path.join(location, self.target_filename), 'w').write('\n'.join(target))


class PatchFile(object):
    def __init__(self, patch):
        self.diffs = []
        lines = patch.split('\n')
        while lines:
            index1 = lines.pop(0)
            assert index1.startswith('Index: ')
            index2 = lines.pop(0)
            assert index2.startswith('==============')
            diff = []
            diff.append(lines.pop(0))
            while lines and not lines[0].startswith('Index: '):
                diff.append(lines.pop(0))
            diff_obj = Diff(diff)
            self.diffs.append(diff_obj)

    def apply(self, location, verbose=True):
        for diff in self.diffs:
            if verbose:
                print("Patching {}".format(diff.source_filename))
            diff.apply(location)


class QuiltSeries(object):
    def __init__(self, patches_location):
        self.patches_location = patches_location
        series = open(os.path.join(self.patches_location, 'series')).read()
        self.patches = []
        self.patch_names = []
        for line in series.split('\n'):
            if not line:
                continue
            patch_content = open(os.path.join(self.patches_location, line)).read()
            self.patches.append(PatchFile(patch_content))
            self.patch_names.append(line)

    def apply(self, location, verbose=True):
        for patch, name in zip(self.patches, self.patch_names):
            if verbose:
                print("Applying {}".format(name))
            patch.apply(location, verbose=verbose)
