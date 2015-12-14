Pysaliency
==========

Pysaliency is a python package for saliency modelling. It aims at providing a unified interface
to both the traditional saliency maps used in saliency modeling as well as probabilistic saliency
models. Pysaliency has a range of influential models prepackaged and ready for use, as well as
some public available datasets. These models are using the original code which is often matlab.
Therefore, a matlab licence is required to make use of these models, although quite some of them
work with octave, too (see below).


Installation
------------

Make sure all packages from requirements.txt are installed. Then as usual, the package is installed by

    python setup.py install

If you want to use the SALICON dataset you have to install the [salicon python api](https://github.com/NUS-VIP/salicon-api).


Quickstart
----------

    import pysaliency
    
    dataset_location = 'datasets'
    model_location = 'models'

    mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=dataset_location)
    aim = pysaliency.AIM(location=model_locations)
    saliency_map = aim.saliency_map(mit_stimuli.stimuli[0])

    plt.imshow(saliency_map)


    auc = aim.AUC(mit_stimuli, mit_fixations)


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
