from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tempfile
import zipfile
from pkg_resources import resource_string,  resource_listdir

from .utils import TemporaryDirectory, download_and_check, run_matlab_cmd
from .quilt import QuiltSeries
from .saliency_map_models import MatlabSaliencyMapModel


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
        super(AIM, self).__init__(os.path.join(self.location, 'AIM_wrapper.m'), only_color_stimuli=True)
        self.filters = filters
        self.convolve = convolve

    def matlab_command(self, stimulus):
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

    def matlab_command(self, stimulus):
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


class ContextAwareSaliency(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    The saliency model "Context-Aware Saliency" by Goferman et. al.
    The original matlab code is used.

    .. seealso::
        Stas Goferman, Lihi Zelnik-Manor, Ayellet Tal. Context-Aware Saliency Detection [CVPR 2010] [PAMI 2012]

        http://webee.technion.ac.il/labs/cgm/Computer-Graphics-Multimedia/Software/Saliency/Saliency.html
    """
    __modelname__ = 'ContextAwareSaliency'

    def __init__(self, location=None):
        self.setup(location)
        super(ContextAwareSaliency, self).__init__(os.path.join(self.location, 'ContextAwareSaliency_wrapper.m'))

    def _setup(self):
        with TemporaryDirectory() as temp_dir:
            if not os.path.isdir(temp_dir):
                os.makedirs(self.location)
            download_and_check('http://webee.technion.ac.il/labs/cgm/Computer-Graphics-Multimedia/Software/Saliency/Saliency.zip',
                               os.path.join(temp_dir, 'Saliency.zip'),
                               'c3c6768ef26e95def76000f51e8aad7c')

            z = zipfile.ZipFile(os.path.join(temp_dir, 'Saliency.zip'))
            source_location = os.path.join(self.location, 'source')
            os.makedirs(source_location)
            z.extractall(source_location)

            with open(os.path.join(self.location, 'ContextAwareSaliency_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/models/ContextAwareSaliency_wrapper.m'))


class BMS(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    "Boolean Map based Saliency (BMS)" by Zhang and Sclaroff.
    The original matlab code is used.

    .. note::
        The original code is slightly patched to work from other directories.
        Also, their code uses OpenCV for performance. The original compile script
        had hardcoded Windows paths for the location of OpenCV. The compile script
        is patched to use the path of OpenCV in Ubuntu. If you want to use the BMS
        model in other systems, you might have to adapt the compile script and run
        it yourself.

        In Debian-based linux systems like Ubuntu you can install all needed
        opencv-libraries with

            apt-get install libopencv-core-dev libopencv-highgui-dev libopencv-imgproc-dev libopencv-flann-dev libopencv-photo-dev libopencv-video-dev libopencv-features2d-dev libopencv-objdetect-dev libopencv-calib3d-dev libopencv-ml-dev libopencv-contrib-dev

    .. seealso::
        Jianming Zhang, Stan Sclaroff. Saliency detection: a boolean map approach [ICCV 2013]
        [http://cs-people.bu.edu/jmzhang/BMS/BMS_iccv13_preprint.pdf]

        http://cs-people.bu.edu/jmzhang/BMS/BMS.html
    """
    __modelname__ = 'BMS'

    def __init__(self, location=None):
        self.setup(location)
        super(BMS, self).__init__(os.path.join(self.location, 'BMS_wrapper.m'))

    def _setup(self):
        with TemporaryDirectory() as temp_dir:
            if not os.path.isdir(temp_dir):
                os.makedirs(self.location)
            download_and_check('http://cs-people.bu.edu/jmzhang/BMS/BMS-mex.zip',
                               os.path.join(temp_dir, 'BMS.zip'),
                               '056fa962993c3083f7977ee18432dd8c')

            z = zipfile.ZipFile(os.path.join(temp_dir, 'BMS.zip'))
            source_location = os.path.join(self.location, 'source')
            os.makedirs(source_location)
            z.extractall(source_location)

            def unpack_directory(package, resource_name, location):
                files = resource_listdir(package, resource_name)
                for file in files:
                    with open(os.path.join(location, file), 'wb') as f:
                        f.write(resource_string(package, os.path.join(resource_name, file)))

            patch_dir = os.path.join(self.location, 'patches')
            os.makedirs(patch_dir)
            unpack_directory(__name__, os.path.join('scripts',
                                                    'models',
                                                    'BMS',
                                                    'patches'),
                             patch_dir)

            series = QuiltSeries(patch_dir)
            series.apply(source_location, verbose=True)

            run_matlab_cmd('compile', cwd=os.path.join(source_location, 'mex'))

            with open(os.path.join(self.location, 'BMS_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/models/BMS/BMS_wrapper.m'))


class GBVS(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    "Graph Based Visual Saliency (GBVS)" by Zhang and Sclaroff.
    The original matlab code is used.

    .. note::
        The original code is slightly patched to work from other directories.
        Also, their code uses OpenCV for performance. The original compile script
        had hardcoded Windows paths for the location of OpenCV. The compile script
        is patched to use the path of OpenCV in Ubuntu. If you want to use the BMS
        model in other systems, you might have to adapt the compile script and run
        it yourself.

    .. seealso::
        Jianming Zhang, Stan Sclaroff. Saliency detection: a boolean map approach [ICCV 2013]
        [http://cs-people.bu.edu/jmzhang/BMS/BMS_iccv13_preprint.pdf]

        http://cs-people.bu.edu/jmzhang/BMS/BMS.html
    """
    __modelname__ = 'GBVS'

    def __init__(self, location=None):
        """
        The parameter explanation have been adopted from the original code.

        Parameters
        ==========

        General
        -------

        @type  salmapmaxsize: integer, defaults to 32
        @param salmapmaxsize: size of calculated saliency maps (maximum dimension).
                              don't set this too high (e.g., >60).

        @type  blurfrac: float, defaults to 0.02
        @param blurfrac: final blur to apply to master saliency map
                         (in standard deviations of gaussian kernel,
                         expressed as fraction of image width).
                         Use value 0 to turn off this feature.

        Features
        --------

        @type  channels: string, defaults to `DIO`.
        @param channels: feature channels to use encoded as a string.
                         these are available:
                           C is for Color
                           I is for Intensity
                           O is for Orientation
                           R is for contRast
                           F is for Flicker
                           M is for Motion
                           D is for DKL Color (Derrington Krauskopf Lennie) ==
                             much better than C channel
                         e.g., 'IR' would be only intensity and
                               contrast, or
                         'CIO' would be only color,int.,ori. (standard)
                         'CIOR' uses col,int,ori, and contrast
        @type  colorWeight, intensityWeight, orientationWeight, contrastWeight, flickerWeight, motionWeight, dklcolorWeight: float, defaults to 1
        @param colorWeight, intensityWeight, orientationWeight, contrastWeight, flickerWeight, motionWeight, dklcolorWeight:
                  weights of feature channels (do not need to sum to 1).

        @type  gaborangles: list, None is the default: [0, 45, 90, 135]
        @param gaborangles: angles of gabor filters

        @type  contrastwidth: float, defaults to 0.1
        @param contrastwidth: fraction of image width = length of square side over which luminance variance is
                              computed for 'contrast' feature map. LARGER values will give SMOOTHER contrast maps.

        @type


        """
        self.setup(location)
        super(GBVS, self).__init__(os.path.join(self.location, 'GBVS_wrapper.m'))

    def _setup(self):
        with TemporaryDirectory() as temp_dir:
            if not os.path.isdir(temp_dir):
                os.makedirs(temp_dir)
            download_and_check('http://www.vision.caltech.edu/~harel/share/gbvs.zip',
                               os.path.join(temp_dir, 'gbvs.zip'),
                               'c5a86b9549c2c0bbd1b7f7e5b663b031')

            z = zipfile.ZipFile(os.path.join(temp_dir, 'gbvs.zip'))
            source_location = os.path.join(self.location, 'gbvs')
            z.extractall(self.location)

            def unpack_directory(package, resource_name, location):
                files = resource_listdir(package, resource_name)
                for file in files:
                    with open(os.path.join(location, file), 'wb') as f:
                        f.write(resource_string(package, os.path.join(resource_name, file)))

            patch_dir = os.path.join(self.location, 'patches')
            os.makedirs(patch_dir)
            unpack_directory(__name__, os.path.join('scripts',
                                                    'models',
                                                    'GBVS',
                                                    'patches'),
                             patch_dir)

            series = QuiltSeries(patch_dir)
            series.apply(source_location, verbose=True)

            #run_matlab_cmd('compile', cwd=os.path.join(source_location, 'mex'))

            with open(os.path.join(self.location, 'GBVS_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/models/GBVS/GBVS_wrapper.m'))
