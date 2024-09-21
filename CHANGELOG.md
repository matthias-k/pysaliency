# Changelog

* 0.3 (dev):
  This is a major release, refactoring most of `pysaliency.datasets` including some breaking changes.
  * Feature: `Scanpaths` is a new class for storing scanpaths. It has a similar API to the old `FixationTrains`,
    but exclusively cares about scanpaths. For example, the length of a Scanpaths instance is the number of scanpaths,
    not the number of fixations. It is intended to make iterating over and working with scanpaths more convenient.
  * Feature: `ScanpathFixations` is a new subclass of `Fixations` intended for handling fixations that come from scanpaths.
    It is intended to replace the old `FixationTrains` class. `ScanpathFixations` has a `scanpaths: Scanpaths`
    attribute storing the source scanpaths (what used to be stored manually as `train_xs` etc in `FixationTrains`).
    Unlike `FixationTrains`, `ScanpathFixations` does not have any attributes that are not derived from the scanpaths.
    `FixationTrains` is now a deprecated subclass of `ScanpathFixations` which adds the old properties and constructor
    and allows for attributes which are neither scanpath attributes nor scanpath fixation attributes.
  * Feature: `VariableLengthArray` for inuititively handling data like scanpaths where each row can have a different length.
    `Fixations.x_Hist`, `Fixations.y_hist`, `Scanpaths.xs` etc are now instances of `VariableLengthArray`.
  * `Fixations.lengths` has been renamed to `Fixations.scanpath_history_length` to make it clear that it is the length of the scanpath history.
    `Fixations.lengths` is now a deprecated alias.
  * In general, naming convention for attriutes has been changed to use the plural form if the attribute is a list of values for each
    element (i.e., `Scanpaths.xs`) and the singular form if the attribute is a single value (i.e., `Scanpaths.length`,  `Fixations.x`). This resulted in
    renaming `Fixations.subjects` to `Fixations.subject`. The old name is now a deprecated alias.
  * Bugfix: Compatibility with torch 2.2 and numpy 2.0
  * Bugfix!: The download location of the RARE2012 model changed. The new source code results in slightly different predictions.
  * Feature: The RARE2007 model is now available as `pysaliency.external_models.RARE2007`. It's execution requires MATLAB.
  * matlab scripts are now called with the `-batch` option instead of `-nodisplay -nosplash -r`, which should behave better.
  * Enhancement: preloaded stimulus ids are passed on to subsets of Stimuli and FileStimuli.


* 0.2.22:
  * Enhancement: New [Tutorial](notebooks/Tutorial.ipynb).
  * Bugfix: `SaliencyMapModel.AUC` failed if some images didn't have any fixations.
  * Feature: `StimulusDependentSaliencyMapModel`
  * Bugfix: The NUSEF dataset scaled some fixations not correctly to image coordinates. Also, we now account for some typos in the
    dataset source data.
  * Feature: CrossvalMultipleRegularizations and GeneralMixtureKernelDensityEstimator in baseline utils (names might change!)
  * Feature: DVAAwareScanpathModel
  * Feature: ShuffledBaselineModel is now much more efficient and able to handle large numbers of stimuli.
    hence, ShuffledSimpleBaselineModel is not necessary anymore and a deprecated alias to ShuffledBaselineModel
  * Feature: ShuffledBaselineModel can now compute predictions for very large numbers of stimuli without needing
    to have all individual predictions in memory due to a recursive reduce logsumexp implementation.
  * Feature: `plotting.plot_scanpath` to visualize scanpaths and saccades. WIP, expect the API to change!
  * Feature: DeepGaze I and DeepGazeIIE models
  * Feature: COCO Freeview dataset
  * Feature: `optimize_for_information_gain(framework='torch', ...) now supports a `cache_directory`,
    where intermediate steps are cached. This supports resuming crashed optimization runs.
  * Bugfix: fixed some edge cases in `optimize_for_information_gain(framework='torch')`
  * Feature: COCO Seach18 dataset
  * Feature: `FixationTrains.train_lengths`
  * Feature: `FixationTrains.scanpath_fixation_attributes` allows handling of per-fixation attributes on scanpath level,
    e.g. fixation durations. According attributes as in a Fixations instance are automatically created,
    e.g. for durations there will be an attribute `durations` and an attribute `duration_hist`. Also
    for scanpath_attributes (e.g., attributes applying to a whole scanpath, such as task) will also generate
    an attribute for each fixation to make this information available in Fixations instance.
  * Feature: `scanpaths_from_fixations` reconstructs a FixationTrains object from a Fixations instance
  * Bugfix: `t_hist` got replaced with `y_hist` in Fixations instances (but luckily not in FixationTrains instances)
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
