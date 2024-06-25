from __future__ import absolute_import, print_function, division, unicode_literals

import os
import zipfile
from pkg_resources import resource_string

from ..utils import TemporaryDirectory, download_and_check, run_matlab_cmd
from ..saliency_map_models import MatlabSaliencyMapModel

from .utils import extract_zipfile, apply_quilt, download_extract_patch, ExternalModelMixin


class AIM(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    The saliency model "Attention based on Information Maximization" (AIM)
    by Bruce and Tsotso. The original matlab code is used.

    .. seealso::
        Bruce, Tsotso: Saliency, attention, and visual search: An information theoretic approach. 2009.

        http://www.cs.umanitoba.ca/~bruce/datacode.html
    """
    __modelname__ = 'AIM'

    def __init__(self, filters='31jade950', convolve=True, location=None, **kwargs):
        self.setup(location)
        super(AIM, self).__init__(os.path.join(self.location, 'AIM_wrapper.m'), only_color_stimuli=True, **kwargs)
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
                               '6d52bc2c0cb15bc186d3d6de32751351',
                               verify_ssl=False)

            z = zipfile.ZipFile(os.path.join(temp_dir, 'AIM.zip'))
            namelist = z.namelist()
            namelist = [n for n in namelist if n.endswith('.m') or n.endswith('.mat')]

            z.extractall(self.location, namelist)
            with open(os.path.join(self.location, 'AIM_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/AIM_wrapper.m'))


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

        This model does not work with octave due to incompabilities
        of octave with matlab. This might change in the future.

    .. seealso::
        Lingyun Zhang, Matthew H. Tong, Tim K. Marks, Honghao Shan, Garrison W. Cottrell. SUN: A Bayesian framework for saliency using natural statistics [JoV 2008]

        http://cseweb.ucsd.edu/~l6zhang/
    """
    __modelname__ = 'SUN'

    def __init__(self, scale=1.0, location=None, **kwargs):
        self.setup(location)
        super(SUN, self).__init__(os.path.join(self.location, 'SUN_wrapper.m'), only_color_stimuli=True, **kwargs)
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
                f.write(resource_string(__name__, 'scripts/SUN_wrapper.m'))
            with open(os.path.join(self.location, 'ensure_image_is_color_image.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/ensure_image_is_color_image.m'))


class ContextAwareSaliency(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    The saliency model "Context-Aware Saliency" by Goferman et. al.
    The original matlab code is used.

    .. seealso::
        Stas Goferman, Lihi Zelnik-Manor, Ayellet Tal. Context-Aware Saliency Detection [CVPR 2010] [PAMI 2012]

        http://webee.technion.ac.il/labs/cgm/Computer-Graphics-Multimedia/Software/Saliency/Saliency.html

        (this URL is down as of 2024-06. Ther doesn't seem to be a new version, but you can check an old version
        at https://web.archive.org/web/20230106184636/https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/Saliency/Saliency.html)
    """
    __modelname__ = 'ContextAwareSaliency'

    def __init__(self, location=None, **kwargs):
        self.setup(location)
        super(ContextAwareSaliency, self).__init__(os.path.join(self.location, 'ContextAwareSaliency_wrapper.m'), **kwargs)

    def _setup(self):
        with TemporaryDirectory() as temp_dir:
            if not os.path.isdir(temp_dir):
                os.makedirs(self.location)
            # url = 'http://webee.technion.ac.il/labs/cgm/Computer-Graphics-Multimedia/Software/Saliency/Saliency.zip'
            # The original URL is not available anymore. We use the webarchive version
            url = 'https://web.archive.org/web/20230106184636/https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/Saliency/Saliency.zip'
            download_and_check(url,
                               os.path.join(temp_dir, 'Saliency.zip'),
                               'c3c6768ef26e95def76000f51e8aad7c')

            z = zipfile.ZipFile(os.path.join(temp_dir, 'Saliency.zip'))
            source_location = os.path.join(self.location, 'source')
            os.makedirs(source_location)
            z.extractall(source_location)

            with open(os.path.join(self.location, 'ContextAwareSaliency_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/ContextAwareSaliency_wrapper.m'))


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

    def __init__(self, location=None, **kwargs):
        self.setup(location)
        super(BMS, self).__init__(os.path.join(self.location, 'BMS_wrapper.m'), **kwargs)

    def _setup(self):
        with TemporaryDirectory() as temp_dir:
            if not os.path.isdir(temp_dir):
                os.makedirs(self.location)
            download_and_check('http://cs-people.bu.edu/jmzhang/BMS/BMS-mex.zip',
                               os.path.join(temp_dir, 'BMS.zip'),
                               '056fa962993c3083f7977ee18432dd8c')

            source_location = os.path.join(self.location, 'source')
            extract_zipfile(os.path.join(temp_dir, 'BMS.zip'),
                            source_location)

            apply_quilt(source_location, __name__, os.path.join('scripts',
                                                                'BMS',
                                                                'patches'),
                        os.path.join(self.location, 'patches'))

            run_matlab_cmd('compile', cwd=os.path.join(source_location, 'mex'))

            with open(os.path.join(self.location, 'BMS_wrapper.m'), 'wb') as f:
                f.write(resource_string(__name__, 'scripts/BMS/BMS_wrapper.m'))


class GBVS(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    "Graph Based Visual Saliency (GBVS)" by Zhang and Sclaroff.
    The original matlab code is used.

    .. note::
        The original code is slightly patched to work from other directories.

        This model does not work with octave due to incompabilities
        of octave with matlab. This might change in the future.

    .. seealso::
        Jonathan Harel, Christof Koch, Pietro Perona. Graph-Based Visual Saliency [NIPS 2006]

        http://www.vision.caltech.edu/~harel/share/gbvs.php
    """
    __modelname__ = 'GBVS'

    def __init__(self, location=None, **kwargs):
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


        """
        self.setup(location)
        super(GBVS, self).__init__(os.path.join(self.location, 'GBVS_wrapper.m'), **kwargs)

    def _setup(self):
        source_location = os.path.join(self.location, 'gbvs')
        download_extract_patch('http://www.vision.caltech.edu/~harel/share/gbvs.zip',
                               'c5a86b9549c2c0bbd1b7f7e5b663b031',
                               source_location,
                               location_in_archive=True,
                               patches=os.path.join('GBVS', 'patches'))

        run_matlab_cmd("addpath('compile');gbvs_compile", cwd=source_location)

        with open(os.path.join(self.location, 'GBVS_wrapper.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/GBVS/GBVS_wrapper.m'))


class GBVSIttiKoch(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    IttiKoch as implemented in "Graph Based Visual Saliency (GBVS)" by Zhang and Sclaroff.
    The original matlab code is used.

    .. note::
        The original code is slightly patched to work from other directories.

        This model does not work with octave due to incompabilities
        of octave with matlab. This might change in the future.

    .. seealso::
        Jonathan Harel, Christof Koch, Pietro Perona. Graph-Based Visual Saliency [NIPS 2006]

        http://www.vision.caltech.edu/~harel/share/gbvs.php
    """
    __modelname__ = 'GBVSIttiKoch'

    def __init__(self, location=None, **kwargs):
        self.setup(location)
        super(GBVSIttiKoch, self).__init__(os.path.join(self.location, 'GBVSIttiKoch_wrapper.m'), **kwargs)

    def _setup(self):
        source_location = os.path.join(self.location, 'gbvs')
        download_extract_patch('http://www.vision.caltech.edu/~harel/share/gbvs.zip',
                               'c5a86b9549c2c0bbd1b7f7e5b663b031',
                               source_location,
                               location_in_archive=True,
                               patches=os.path.join('GBVS', 'patches'))

        run_matlab_cmd("addpath('compile');gbvs_compile", cwd=source_location)

        with open(os.path.join(self.location, 'GBVSIttiKoch_wrapper.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/GBVS/GBVSIttiKoch_wrapper.m'))


class Judd(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    Judd model by Judd et al.
    The original matlab code is used.

    .. note::
        The original code is patched to work from other directories.

        The model makes use of the [SaliencyToolbox](http://www.saliencytoolbox.net/). Due
        to licence restrictions the Toolbox cannot be downloaded automatically. You have to
        download it yourself and provide the location of the zipfile via the
        `saliency_toolbox_archive`-keyword to the constructor.

        This model does not work with octave due to incompabilities
        of octave with matlab. This might change in the future.

    .. seealso::
        Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to predict where humans look [ICCV 2009]

        http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
    """
    __modelname__ = 'Judd'

    def __init__(self, location=None, saliency_toolbox_archive=None, include_locations=None, library_locations=None, **kwargs):
        self.setup(location, saliency_toolbox_archive=saliency_toolbox_archive, include_locations=include_locations, library_locations=library_locations)
        super(Judd, self).__init__(os.path.join(self.location, 'Judd_wrapper.m'), only_color_stimuli=True, **kwargs)

    def _setup(self, saliency_toolbox_archive, include_locations, library_locations):
        if not saliency_toolbox_archive:
            raise Exception('You have to provide the zipfile containing the Itti and Koch Saliency Toolbox!')

        if include_locations is None:
            include_locations = ['/usr/include/opencv2']
        if library_locations is None:
            library_locations = ['/usr/lib/x86_64-linux-gnu/']

        source_location = os.path.join(self.location, 'source')
        print('Downloading Judd Model...')
        download_extract_patch('http://people.csail.mit.edu/tjudd/WherePeopleLook/Code/JuddSaliencyModel.zip',
                               '03e56c6f37c0ef3605c4c476f8a35b6b',
                               os.path.join(source_location, 'JuddSaliencyModel'),
                               location_in_archive=True,
                               patches=os.path.join('Judd', 'JuddSaliencyModel_patches'))

        print('Downloading matlabPyrTools...')
        download_extract_patch('http://www.cns.nyu.edu/pub/eero/matlabPyrTools.tar.gz',
                               'da53c02c843ed636bb35e20dac68a1b2',
                               os.path.join(source_location, 'matlabPyrTools'),
                               location_in_archive=True,
                               patches=None)

        print('Downloading voc')
        download_extract_patch('http://cs.brown.edu/~pff/latent-release3/voc-release3.1.tgz',
                               '20502f8a40f1122e00f81dcc0d11a843',
                               os.path.join(source_location, 'voc-release3.1'),
                               location_in_archive=True,
                               patches=os.path.join('Judd', 'voc_patches'),
                               verify_ssl=False,  # doesn't seem to support SSL
                               )
        run_matlab_cmd("compile;quit;", cwd=os.path.join(source_location, 'voc-release3.1'))

        print('Extracting Saliency Toolbox')
        extract_zipfile(saliency_toolbox_archive, source_location)
        apply_quilt(os.path.join(source_location, 'SaliencyToolbox'),
                    __name__, os.path.join('scripts', 'Judd', 'SaliencyToolbox_patches'),
                    os.path.join(source_location, 'SaliencyToolbox_patches'))

        print('Downloading Viola Jones Face Detection')
        download_extract_patch('http://www.mathworks.com/matlabcentral/fileexchange/19912-open-cv-viola-jones-face-detection-in-matlab?download=true',
                               '26d3f6bc6641d959661afedadff8b479',
                               os.path.join(source_location, 'FaceDetect'),
                               location_in_archive=False,
                               patches=os.path.join('Judd', 'FaceDetect_patches'))

        includes = ' '.join(['-I{}'.format(location) for location in include_locations])
        libraries = ' '.join(['-L{}'.format(location) for location in library_locations])


        run_matlab_cmd("mex FaceDetect.cpp {includes} {libaries} -lopencv_core -lopencv_objdetect -lopencv_imgproc -lopencv_highgui -outdir .".format(
                           includes=includes,
                           libaries=libraries
                       ),
                       cwd=os.path.join(source_location, 'FaceDetect', 'src'))

        print('Downloading LabelMe Toolbox')
        download_extract_patch('http://labelme.csail.mit.edu/LabelMeToolbox/LabelMeToolbox.zip',
                               '66d982aae539149eff09ce26b7278f4d',
                               os.path.join(source_location, 'LabelMeToolbox'),
                               location_in_archive=True,
                               patches=None)

        with open(os.path.join(self.location, 'Judd_wrapper.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/Judd/Judd_wrapper.m'))


class IttiKoch(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    Model by Itti and Koch.
    The original matlab code is used.

    .. note::
        The original code is patched to work from other directories.

        The model makes use of the [SaliencyToolbox](http://www.saliencytoolbox.net/). Due
        to licence restrictions the Toolbox cannot be downloaded automatically. You have to
        download it yourself and provide the location of the zipfile via the
        `saliency_toolbox_archive`-keyword to the constructor.

        This model does not work with octave due to incompabilities
        of octave with matlab. This might change in the future.

    .. seealso::
        Saliency Toolbox by: Dirk Walther, Christof Koch. Modeling attention to salient proto-objects [Neural Networks 2006]

        http://www.saliencytoolbox.net/
    """
    __modelname__ = 'IttiKoch'

    def __init__(self, location=None, saliency_toolbox_archive=None, **kwargs):
        self.setup(location, saliency_toolbox_archive=saliency_toolbox_archive)
        super(IttiKoch, self).__init__(os.path.join(self.location, 'IttiKoch_wrapper.m'), only_color_stimuli=True, **kwargs)

    def _setup(self, saliency_toolbox_archive):
        if not saliency_toolbox_archive:
            raise Exception('You have to provide the zipfile containing the Itti and Koch Saliency Toolbox!')

        print('Extracting Saliency Toolbox')
        extract_zipfile(saliency_toolbox_archive, self.location)
        apply_quilt(os.path.join(self.location, 'SaliencyToolbox'),
                    __name__, os.path.join('scripts', 'Judd', 'SaliencyToolbox_patches'),
                    os.path.join(self.location, 'SaliencyToolbox_patches'))

        with open(os.path.join(self.location, 'IttiKoch_wrapper.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/IttiKoch_wrapper.m'))


class RARE2007(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    RARE2007 by Mancas.
    The original matlab code is used.

    .. note::
        The source code published by the authors is used without modifications.

    .. seealso::
        Matei Mancas. Relative influence of bottom-up and top-down attention.
        [International Workshop on Attention in Cognitive Systems 2008]

        https://numediart.github.io/VisualAttention-RareFamily/
        https://github.com/numediart/VisualAttention-Rare2007
    """
    __modelname__ = 'RARE2007'

    def __init__(self, location=None, **kwargs):
        self.setup(location)
        super(RARE2007, self).__init__(os.path.join(self.location, 'RARE2007_wrapper.m'), only_color_stimuli=True, **kwargs)

    def _setup(self):
        source_location = os.path.join(self.location, 'source')
        directory = 'VisualAttention-Rare2007-ee7ef25c9b6cc7e96837091e868d03bd97b4b131'
        print('Downloading RARE2007 Model...')
        url = 'https://github.com/numediart/VisualAttention-Rare2007/archive/ee7ef25c9b6cc7e96837091e868d03bd97b4b131.zip'
        download_extract_patch(url,
                               '7ee05ae059dac10478b5cabc8ee066dc',
                               os.path.join(source_location, directory),
                               location_in_archive=True,
                               patches=None)

        with open(os.path.join(self.location, 'RARE2007_wrapper.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/RARE2007_wrapper.m'))


class RARE2012(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    RARE2012 by Margolin et al.
    The original matlab code is used.

    .. note::


        This page is not available anymore, the RARE models are now hosted
        at https://numediart.github.io/VisualAttention-RareFamily/, the RARE2012
        model is published at https://github.com/numediart/VisualAttention-Rare2012.
        This model is a wrapper around the published (compiled) matlab code.

        The model used to be hosted at
        http://www.tcts.fpms.ac.be/attention/?categorie16/what-and-why.
        This page is not available anymore, the RARE models are now hosted
        at https://numediart.github.io/VisualAttention-RareFamily/ and GitHub.
        With this change, the behaviour of RARE2012 changed slightly
        compared to the earlier version (e.g., saliency maps look a bit smoother).

        This model does not work with octave due to incompabilities
        of octave with matlab (RARE2012 comes as precompiled matlab file).
        This might change in the future.

    .. seealso::
        Nicolas Riche, Matei Mancas, Matthieu Duvinage, Makiese Mibulumukini,
        Bernard Gosselin, Thierry Dutoit. RARE2012: A multi-scale rarity-based
        saliency detection with its comparative statistical analysis
        [Signal Processing: Image Communication, 2013]

        https://numediart.github.io/VisualAttention-RareFamily/
    """
    __modelname__ = 'RARE2012'

    def __init__(self, location=None, **kwargs):
        self.setup(location)
        super(RARE2012, self).__init__(os.path.join(self.location, 'RARE2012_wrapper.m'), only_color_stimuli=True, **kwargs)

    def _setup(self):
        source_location = os.path.join(self.location, 'source')
        directory = 'VisualAttention-Rare2012-55ba7414b971429e5e899ddfa574e4235fc806e6'
        print('Downloading RARE2012 Model...')
        url = 'https://github.com/numediart/VisualAttention-Rare2012/archive/55ba7414b971429e5e899ddfa574e4235fc806e6.zip'
        download_extract_patch(url,
                               'a5bbd8a42c231530834659acfd0d2b64.',
                               os.path.join(source_location, directory),
                               location_in_archive=True,
                               patches=None)

        with open(os.path.join(self.location, 'RARE2012_wrapper.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/RARE2012_wrapper.m'))


class CovSal(ExternalModelMixin, MatlabSaliencyMapModel):
    """
    CovSal by Erdem and Erdem
    The original matlab code is used.

    .. note::
        This model does not work with octave due to incompabilities
        of octave with matlab. This might change in the future.

    .. seealso::
        Erkut Erdem, Aykut Erdem. Visual saliency estimation by nonlinearly integrating features using region covariances [JoV 2013]

        http://web.cs.hacettepe.edu.tr/~erkut/projects/CovSal/
    """
    __modelname__ = 'CovSal'

    def __init__(self, location=None, size=512, quantile=0.1, centerbias=True, modeltype='SigmaPoints', **kwargs):
        """
        Parameters
        ----------

        @type  size: int
        @param size: size of rescaled image

        @type  quantile: float
        @param quantile: parameter specifying the most similar regions in the neighborhood

        @type  centerbias: bool
        @param centerbias: True for center bias and False for no center bias

        @type  modeltype: string
        @param modeltype: 'CovariancesOnly' and 'SigmaPoints' to denote whether
                          first-order statistics will be incorporated or not

        """
        self.setup(location)
        super(CovSal, self).__init__(os.path.join(self.location, 'CovSal_wrapper.m'), only_color_stimuli=True, **kwargs)
        self.size = size
        self.quantile = quantile
        self.centerbias = centerbias
        if modeltype not in ['CovariancesOnly', 'SigmaPoints']:
            raise ValueError('Unkown modeltype', modeltype)
        self.modeltype = modeltype

    def matlab_command(self, stimulus):
        return "CovSal_wrapper('{{stimulus}}', '{{saliency_map}}', {}, {}, {}, '{}');".format(self.size,
                                                                                              self.quantile,
                                                                                              1 if self.centerbias else 0,
                                                                                              self.modeltype)

    def _setup(self):
        download_extract_patch('https://web.cs.hacettepe.edu.tr/~erkut/projects/CovSal/saliency.zip',
                               '5c1bcdce438a08051968b085f72b2fdc',
                               os.path.join(self.location, 'saliency'),
                               location_in_archive=True,
                               patches=None)

        with open(os.path.join(self.location, 'CovSal_wrapper.m'), 'wb') as f:
            f.write(resource_string(__name__, 'scripts/CovSal_wrapper.m'))
