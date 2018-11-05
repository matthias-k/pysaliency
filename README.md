Pysaliency
==========

Pysaliency is a python package for saliency modelling. It aims at providing a unified interface
to both the traditional saliency maps used in saliency modeling as well as probabilistic saliency
models.

Pysaliency can evaluate most commonly used saliency metrics, including AUC, sAUC, NSS, CC
image-based KL divergence, fixation based KL divergence and SIM for saliency map models and
log likelihoods and information gain for probabilistic models.

Pysaliency provides several important datasets:

* MIT1003
* MIT300
* CAT2000
* Toronto
* Koehler
* iSUN
* SALICON (both the 2015 and the 2017 edition and each with both the original mouse traces and the inferred fixations)
* FIGRIM
* OSIE
* NUSEF (the part with public images)

and some influential models:
* AIM
* SUN
* ContextAwareSaliency
* BMS
* GBVS
* GBVSIttiKoch
* Judd
* IttiKoch
* RARE2012
* CovSal


These models are using the original code which is often matlab.
Therefore, a matlab licence is required to make use of these models, although quite some of them
work with octave, too (see below).


Installation
------------

You can install pysaliency from pypi via

    pip install pysaliency


Quickstart
----------

    import pysaliency
    
    dataset_location = 'datasets'
    model_location = 'models'

    mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=dataset_location)
    aim = pysaliency.AIM(location=model_location)
    saliency_map = aim.saliency_map(mit_stimuli.stimuli[0])

    plt.imshow(saliency_map)


    auc = aim.AUC(mit_stimuli, mit_fixations)

If you already have saliency maps for some dataset, you can import them into pysaliency easily:

    my_model = pysaliency.SaliencyMapModelFromDirectory(mit_stimuli, '/path/to/my/saliency_maps')
    auc = my_model.AUC(mit_stimuli, mit_fixations)


Using Octave
------------

pysaliency will fall back to octave if no matlab is installed.
Some models might work with octave, e.g. AIM and GBVSIttiKoch. In Debian/Ubuntu you need to install
`octave`, `octave-image`, `octave-statistics`, `liboctave-dev`.

These models and dataset seem to work with octave:

- models
  - AIM
  - GBVSIttiKoch
- datasets
  - Toronto
  - MIT1003
  - MIT300
  - SALICON

Dependencies
-----------

The Judd Model needs some libraries to work. In ubuntu/debian you need to install these packages:
`libopencv-core-dev, libopencv-flann-dev, libopencv-imgproc-dev, libopencv-photo-dev, libopencv-video-dev, libopencv-features2d-dev, libopencv-objdetect-dev, libopencv-calib3d-dev, libopencv-ml-dev, opencv2/contrib/contrib.hpp`
