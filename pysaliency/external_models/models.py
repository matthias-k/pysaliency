from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tempfile
import zipfile
import tarfile
from importlib.resources import files as _resource_files


def resource_string(package, resource):
    return _resource_files(package).joinpath(resource).read_bytes()


def resource_listdir(package, resource_name):
    return [r.name for r in _resource_files(package).joinpath(resource_name).iterdir()]

from boltons.fileutils import mkdir_p
import numpy as np
from scipy.ndimage import zoom

from ..utils import download_and_check, run_matlab_cmd
from ..quilt import QuiltSeries
from ..saliency_map_models import MatlabSaliencyMapModel, SaliencyMapModel

from .utils import write_file, extract_zipfile, unpack_directory, apply_quilt, download_extract_patch, ExternalModelMixin