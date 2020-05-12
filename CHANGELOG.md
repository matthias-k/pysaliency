# Changelog

* 0.2.20 (unpublished):
  * Stimuli now support attributes, just like Fixations. The CAT2000 train and test
    datasets now have the stimulus categories as attribute.
  * failure to download and setup a dataset will no longer result in leftover
    dataset files that keep pysaliency from trying again.
  * crossvalidation splits now support stratifying stimulus attributes
  * the MIT1003 dataset now also contains the history of fixation durations
  * FixationIndexDependentModel
  * Bugfix: The CC of a constant saliency map wrt to a nonconstant one
    now returns zero (instead of nan as previously).
* 0.2.19:
  * added pytorch implementation for optimization of similarity metric as alternative
    to tensorflow implementation which still uses tensorflow 1.x
  * added pytorch implementation for saliency map processing as alternative
    to theano implementation.
  * removed obsolete dependency on openmp
  * made import of pytorch, theano and tensorflow optional
  * bugfixes in precomputed models for stimuli sets with nested directories
