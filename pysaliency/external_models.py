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
    by Bruce and Tsotso.

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
