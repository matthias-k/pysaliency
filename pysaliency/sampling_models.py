from abc import ABC, abstractmethod


class SamplingModelMixin(ABC):
    """A sampling model is supports sampling fixations and whole scanpaths."""
    def sample_scanpath(
        self, stimulus, x_hist, y_hist, t_hist, samples, verbose=False, rst=None
    ):
        """return xs, ys, ts"""
        xs = list(x_hist)
        ys = list(y_hist)
        ts = list(t_hist)
        for i in range(samples):
            x, y, t = self.sample_fixation(stimulus, xs, ys, ts, verbose=verbose, rst=rst)
            xs.append(x)
            ys.append(y)
            ts.append(t)

        return xs, ys, ts

    @abstractmethod
    def sample_fixation(self, stimulus, x_hist, y_hist, t_hist, verbose=False, rst=None):
        """return x, y, t"""
        raise NotImplementedError()
