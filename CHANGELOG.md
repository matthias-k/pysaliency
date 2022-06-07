# Changelog

* 0.2.22 (dev):
  * Bugfix: torch code was broken due to changes in torch 1.11
  * Bugfix: SALICON dataset download did not work anymore
  * Bugfix: NUSEF datast links changed

* 0.2.21:
  * Added new datasets: PASCAL-S and DUT-OMRON
  * Feature: FixedStimulusSizeModel and DVAAwareModel
  * Feature: Fixations finally support len()
  * Experimental feature: conditional_log_densities(stimuli, fixations) and conditional_saliency_maps(...).
    This is WIP to enable batch processing in models.
  * Fallback models for stimulus dependent models
  * MixtureScanpathModel
  * Reimplemented AUC for special case of only one positive sample, leading to substantial speedup
  * There is a new version of the CAT2000 train dataset which fixes some details in the processing.
    Since it changes the dataset, by default the old processing is used.
  * Feature: ShuffledSimpleBaselineModel. Baseline model to be used with ShuffledAUCSaliencyMapModel
    in cases where using ShuffledBaselineModel is not feasible.
  * `pysaliency.get_toronto` now returns a `Fixations` instance instead of `FixationTrains` since
    we don not have scanpath information.
  * `pysaliency.baseline_utils.KDEGoldModel` now supports a keyword argument `grid_spacing` which
    controls how densly the log density of the KDEModel is computed before it is linearly interpolated.
    This can substantially speed up computations on high resolution images.
  * Feature: `pysaliency.precomputed_models.SaliencyMapModelFromArchive` and `ModelFromArchive`
    for loading model predictions from ZIP, TAR and RAR files.
  * Bugfix: all matlab scripts where missing in the pip installation since the change
    to setuptools.
* 0.2.20:
  * Stimuli now support attributes, just like Fixations. The CAT2000 train and test
    datasets now have the stimulus categories as attribute.
  * failure to download and setup a dataset will no longer result in leftover
    dataset files that keep pysaliency from trying again.
  * crossvalidation splits now support stratifying stimulus attributes
  * the MIT1003 dataset now also contains the history of fixation durations
  * FixationIndexDependentModel
  * Bugfix: The CC of a constant saliency map wrt to a nonconstant one
    now returns zero (instead of nan as previously).
  * Feature: Added keyword argument `attributes` to `Fixations` constructor
  * Feature: Provide KLDiv and SIM as functions that can be applied to saliency maps without need for a model.
* 0.2.19:
  * added pytorch implementation for optimization of similarity metric as alternative
    to tensorflow implementation which still uses tensorflow 1.x
  * added pytorch implementation for saliency map processing as alternative
    to theano implementation.
  * removed obsolete dependency on openmp
  * made import of pytorch, theano and tensorflow optional
  * bugfixes in precomputed models for stimuli sets with nested directories
