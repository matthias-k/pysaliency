from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tempfile
import zipfile
import tarfile
from pkg_resources import resource_string,  resource_listdir

from boltons.fileutils import mkdir_p
import numpy as np
from scipy.ndimage import zoom

from ..utils import TemporaryDirectory, download_and_check, run_matlab_cmd
from ..quilt import QuiltSeries
from ..saliency_map_models import MatlabSaliencyMapModel, SaliencyMapModel

from .utils import write_file, extract_zipfile, unpack_directory, apply_quilt, download_extract_patch, ExternalModelMixin