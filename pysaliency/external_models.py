from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tempfile
import zipfile
from pkg_resources import resource_string

from .utils import TemporaryDirectory, download_and_check
from .models import MatlabSaliencyMapModel


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
    def setup(self, location):
        if location is None:
            self.location = tempfile.mkdtemp()
            self._setup()
        else:
            self.location = os.path.join(location, self.__modelname__)
            if not os.path.exists(self.location):
                self._setup()

    def _setup(self):
        raise NotImplementedError()


class AIM(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    The saliency model "Attention based on Information Maximization" (AIM)
    by Bruce and Tsotso. The original matlab code is used.

    .. seealso::
        Bruce, Tsotso: Saliency, attention, and visual search: An information theoretic approach. 2009.

        http://www.cs.umanitoba.ca/~bruce/datacode.html
    """
    __modelname__ = 'AIM'

    def __init__(self, filters='31jade950', convolve=True, location=None):
        self.setup(location)
        super(AIM, self).__init__(os.path.join(self.location, 'AIM_wrapper.m'))
        self.filters = filters
        self.convolve = convolve

    @property
    def matlab_command(self):
        return "AIM_wrapper('{{stimulus}}', '{{saliency_map}}', {}, '{}');".format(1 if self.convolve else 0,
                                                                                   self.filters)

    def _setup(self):
        with TemporaryDirectory() as temp_dir:
            if not os.path.isdir(temp_dir):
                os.makedirs(self.location)
            download_and_check('http://www.cs.umanitoba.ca/~bruce/AIM.zip',
                               os.path.join(temp_dir, 'AIM.zip'),
                               '6d52bc2c0cb15bc186d3d6de32751351')

            z = zipfile.ZipFile(os.path.join(temp_dir, 'AIM.zip'))
            namelist = z.namelist()
            namelist = [n for n in namelist if n.endswith('.m') or n.endswith('.mat')]

            z.extractall(self.location, namelist)
            with open(os.path.join(self.location, 'AIM_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/models/AIM_wrapper.m'))


class SUN(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    The saliency model "Saliency using natural image statistics" (SUN)
    by Zhang et al. The original matlab code is used.

    .. note::
        Due to the patch based approach, the original code
        returns a saliency map that is smaller than the original
        stimulus. Here, in a wrapper script, we enlargen the
        saliency map to match the original stimulus by setting the
        boundary area to the minimal saliency value, as is
        done also by Bruce and Tsotso in AIM.

        As the original code can handle only color images,
        we convert grayscale images to color images by setting
        all color channels to the grayscale value.

    .. seealso::
        Lingyun Zhang, Matthew H. Tong, Tim K. Marks, Honghao Shan, Garrison W. Cottrell. SUN: A Bayesian framework for saliency using natural statistics [JoV 2008]

        http://cseweb.ucsd.edu/~l6zhang/
    """
    __modelname__ = 'SUN'

    def __init__(self, scale=1.0, location=None):
        self.setup(location)
        super(SUN, self).__init__(os.path.join(self.location, 'SUN_wrapper.m'))
        self.scale = scale

    @property
    def matlab_command(self):
        return "SUN_wrapper('{{stimulus}}', '{{saliency_map}}', {});".format(self.scale)

    def _setup(self):
        with TemporaryDirectory() as temp_dir:
            if not os.path.isdir(temp_dir):
                os.makedirs(self.location)
            download_and_check('http://cseweb.ucsd.edu/~l6zhang/code/imagesaliency.zip',
                               os.path.join(temp_dir, 'SUN.zip'),
                               'df69e6c34b2e9e5ddd7a051d98e880d0')

            z = zipfile.ZipFile(os.path.join(temp_dir, 'SUN.zip'))
            namelist = z.namelist()
            namelist = [n for n in namelist if n.endswith('.m') or n.endswith('.mat')]

            z.extractall(self.location, namelist)
            with open(os.path.join(self.location, 'SUN_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/models/SUN_wrapper.m'))
            with open(os.path.join(self.location, 'ensure_image_is_color_image.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/ensure_image_is_color_image.m'))
