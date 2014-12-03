from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tempfile
import zipfile
from pkg_resources import resource_string

from .utils import TemporaryDirectory, download_and_check
from .models import MatlabSaliencyMapModel


class AIM(MatlabSaliencyMapModel):
    def __init__(self, filters='31jade950', convolve=True, location=None):
        if location is None:
            self.location = tempfile.mkdtemp()
            self.setup()
        else:
            self.location = os.path.join(location, 'AIM')
            if not os.path.exists(self.location):
                self.setup()
        super(AIM, self).__init__(os.path.join(self.location, 'AIM_wrapper.m'))

    def setup(self):
        with TemporaryDirectory() as temp_dir:
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
