import numpy as np
import sys
from typing import Union
from scipy.ndimage import gaussian_filter
import pysaliency


class LocalContrastModel(pysaliency.Model):
    def __init__(self, bandwidth=0.05, **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        
    def _log_density(self, stimulus: Union[pysaliency.datasets.Stimulus, np.ndarray]):
        
        # _log_density can either take pysaliency Stimulus objects, or, for convenience, simply numpy arrays
        # `as_stimulus` ensures that we have a Stimulus object
        stimulus_object = pysaliency.datasets.as_stimulus(stimulus)

        # grayscale image
        gray_stimulus = np.mean(stimulus_object.stimulus_data, axis=2)

        # size contains the height and width of the image, but not potential color channels
        height, width = stimulus_object.size

        # define kernel size based on image size
        kernel_size = np.round(self.bandwidth * max(width, height)).astype(int)
        sigma = (kernel_size - 1) / 6
            
        # apply Gausian blur and calculate squared difference between blurred and original image
        blurred_stimulus = gaussian_filter(gray_stimulus, sigma)

        prediction = gaussian_filter((gray_stimulus - blurred_stimulus)**2, sigma)

        # normalize to [1, 255]
        prediction = (254 * (prediction / prediction.max())).astype(int) + 1

        density = prediction / prediction.sum()
        
        return np.log(density)
    
class MySimpleScanpathModel(pysaliency.ScanpathModel):
    def __init__(self, spatial_model_bandwidth: float=0.05, saccade_width: float=0.1):
        self.spatial_model_bandwidth = spatial_model_bandwidth
        self.saccade_width = saccade_width
        self.spatial_model = LocalContrastModel(spatial_model_bandwidth)
        # self.spatial_model = pysaliency.UniformModel()


    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None,):
        stimulus_object = pysaliency.datasets.as_stimulus(stimulus)

        # size contains the height and width of the image, but not potential color channels
        height, width = stimulus_object.size

        spatial_prior_log_density = self.spatial_model.log_density(stimulus)
        spatial_prior_density = np.exp(spatial_prior_log_density)

        # compute saccade bias
        last_x = x_hist[-1]
        last_y = y_hist[-1]
        
        xs = np.arange(width, dtype=float)
        ys = np.arange(height, dtype=float)
        XS, YS = np.meshgrid(xs, ys)

        XS -= last_x
        YS -= last_y
        
        # compute prior
        max_size = max(width, height)
        actual_kernel_size = self.saccade_width * max_size

        saccade_bias = np.exp(-0.5 * (XS ** 2 + YS ** 2) / actual_kernel_size ** 2)

        prediction = spatial_prior_density * saccade_bias
        
        density = prediction / prediction.sum()
        return np.log(density)

