from __future__ import print_function, absolute_import, division, unicode_literals

from abc import ABCMeta, abstractmethod
from six import add_metaclass

from .utils import remove_trailing_nans


@add_metaclass(ABCMeta)
class SamplingModelMixin(object):
    """A sampling model is supports sampling fixations and whole scanpaths."""
    def sample_scanpath(
        self, stimulus, x_hist, y_hist, t_hist, samples, attributes=None, verbose=False, rst=None
    ):
        """return xs, ys, ts"""
        xs = list(remove_trailing_nans(x_hist))
        ys = list(remove_trailing_nans(y_hist))
        ts = list(remove_trailing_nans(t_hist))
        if not len(xs) == len(ys) == len(ts):
            raise ValueError("Histories for x, y and t have to be the same length")

        for i in range(samples):
            x, y, t = self.sample_fixation(stimulus, xs, ys, ts, attributes=attributes, verbose=verbose, rst=rst)
            xs.append(x)
            ys.append(y)
            ts.append(t)

        return xs, ys, ts

    @abstractmethod
    def sample_fixation(self, stimulus, x_hist, y_hist, t_hist, attributes=None, verbose=False, rst=None):
        """return x, y, t"""
        raise NotImplementedError()


class ScanpathSamplingModelMixin(SamplingModelMixin):
    """A sampling model which only has to implement sample_scanpath instead of sample_fixation"""
    @abstractmethod
    def sample_scanpath(
            self, stimulus, x_hist, y_hist, t_hist, samples, attributes=None, verbose=False, rst=None
    ):
        raise NotImplementedError()

    def sample_fixation(self, stimulus, x_hist, y_hist, t_hist, attributes=None, verbose=False, rst=None):
        samples = 1
        xs, ys, ts = self.sample_scanpath(stimulus, x_hist, y_hist, t_hist, samples, attributes=attributes,
                                          verbose=verbose, rst=rst)
        return xs[-1], ys[-1], ts[-1]
