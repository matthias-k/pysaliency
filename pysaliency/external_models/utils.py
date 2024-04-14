from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tempfile
import zipfile
import tarfile
from pkg_resources import resource_string,  resource_listdir

from ..utils import TemporaryDirectory, download_and_check
from ..quilt import QuiltSeries


def write_file(filename, contents):
    """Write contents to file and close file savely"""
    with open(filename, 'wb') as f:
        f.write(contents)


def extract_zipfile(filename, extract_to):
    if zipfile.is_zipfile(filename):
        z = zipfile.ZipFile(filename)
        #os.makedirs(extract_to)
        z.extractall(extract_to)
    elif tarfile.is_tarfile(filename):
        t = tarfile.open(filename)
        t.extractall(extract_to)
    else:
        raise ValueError('Unkown archive type', filename)


def unpack_directory(package, resource_name, location):
    files = resource_listdir(package, resource_name)
    for file in files:
        write_file(os.path.join(location, file),
                   resource_string(package, os.path.join(resource_name, file)))


def apply_quilt(source_location, package, resource_name, patch_directory, verbose=True):
    """Apply quilt series from package data to source code"""
    os.makedirs(patch_directory)
    unpack_directory(package, resource_name, patch_directory)
    series = QuiltSeries(patch_directory)
    series.apply(source_location, verbose=verbose)


def download_extract_patch(url, hash, location, location_in_archive=True, patches=None, verify_ssl=True):
    """Download, extract and maybe patch code"""
    with TemporaryDirectory() as temp_dir:
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        archive_name = os.path.basename(url)
        download_and_check(url,
                           os.path.join(temp_dir, archive_name),
                           hash,
                           verify_ssl=verify_ssl)

        if location_in_archive:
            target = os.path.dirname(os.path.normpath(location))
        else:
            target = location
        extract_zipfile(os.path.join(temp_dir, archive_name),
                        target)

    if patches:
        parent_directory = os.path.dirname(os.path.normpath(location))
        patch_directory = os.path.join(parent_directory, os.path.basename(patches))
        apply_quilt(location, __name__,  os.path.join('scripts', patches), patch_directory)


class ExternalModelMixin(object):
    """
    Download and cache necessary files.

    If the location is None, a temporary directory will be used.
    If the location is not None, the data will be stored in a
    subdirectory of location named after `__modelname`. If this
    sub directory already exists, the initialization will
    not be run.

    After running `setup()`, the actual location will be
    stored in `self.location`.

    To make use of this Mixin, overwrite `_setup()`
    and run `setup(location)`.
    """
    def setup(self, location, *args, **kwargs):
        if location is None:
            self.location = tempfile.mkdtemp()
            self._setup(*args, **kwargs)
        else:
            self.location = os.path.join(location, self.__modelname__)
            if not os.path.exists(self.location):
                self._setup(*args, **kwargs)

    def _setup(self, *args, **kwargs):
        raise NotImplementedError()