from __future__ import absolute_import, print_function, division#, unicode_literals

import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse


def nonlinearity(input, x, y, length):
    """
    Apply a pointwise nonlinearity to input

    The nonlinearity is a picewise linear function.
    The graph of the function is given by the vectors x and y.
    """
    parts = []
    for i in range(length-1):
        x1 = x[i]
        x2 = x[i+1]
        y1 = y[i]
        y2 = y[i+1]
        #print x1.tag
        part = (y2-y1)/(x2-x1)*(theano.tensor.clip(input, x1, x2)-x1)
        parts.append(part)
    output = y[0]
    for part in parts:
        output = output + part
    return output


def gaussian_filter(input, sigma, window_radius = 40):
    """
    Filter input with a Gaussian using mode `nearest`.

    input is expected to be three dimensional of type n times x times y
    """

    # Construction of 1d kernel
    #filter_1d = T.arange(-window_radius, window_radius+1)
    # Work around some strange theano bug
    filter_1d = T.arange(2*window_radius + 1) - window_radius

    filter_1d = T.exp(-0.5*filter_1d**2/sigma**2)
    filter_1d = filter_1d / filter_1d.sum()
    filter_1d = filter_1d.astype(input.dtype)

    W = filter_1d.dimshuffle([0, 'x'])
    W2 = filter_1d.dimshuffle(['x', 0])

    blur_input = input.dimshuffle(['x', 0, 1, 2])
    filter_W = W.dimshuffle(['x', 'x', 0, 1])
    filter_W2 = W2.dimshuffle(['x', 'x', 0, 1])

    # Construction of filter pipeline
    blur_input_start = blur_input[:, :, :1, :]
    blur_input_end = blur_input[:, :, -1:, :]
    padded_input = T.concatenate([blur_input_start]*window_radius+[blur_input]+[blur_input_end]*window_radius, axis=2)

    blur_op = T.nnet.conv2d(padded_input, filter_W, border_mode='valid', filter_shape=[1, 1, None, None])
    #x_min = (W.shape[1]-1)//2
    #x_max = input.shape[2]+(W.shape[1]-1)//2
    #y_min = (W.shape[0]-1)//2+window_radius
    #y_max = input.shape[1]+(W.shape[0]-1)//2+window_radius
    #cropped_output1 = blur_op[:, :, y_min:y_max, x_min:x_max]
    #cropped_output1_start = blur_op[:, :, y_min:y_max, x_min:x_min+1]
    #cropped_output1_end = blur_op[:, :, y_min:y_max, x_max-1:x_max]

    cropped_output1 = blur_op
    cropped_output1_start = blur_op[:, :, :, :1]
    cropped_output1_end = blur_op[:, :, :, -1:]
    padded_cropped_input = T.concatenate([cropped_output1_start]*window_radius
                                         + [cropped_output1]
                                         + [cropped_output1_end] * window_radius, axis=3)
    blur_op2 = T.nnet.conv2d(padded_cropped_input, filter_W2, border_mode='valid', filter_shape=[1, 1, None, None])
    #x_min2 = (W2.shape[1]-1)//2+window_radius
    #x_max2 = input.shape[2]+(W2.shape[1]-1)//2+window_radius
    #y_min2 = (W2.shape[0]-1)//2
    #y_max2 = input.shape[1]+(W2.shape[0]-1)//2
    cropped_output2 = blur_op2[0, :, :, :]  # [0, :, y_min2:y_max2, x_min2:x_max2]
    return cropped_output2


class Blur(object):
    def __init__(self, input, sigma=20.0, window_radius=60):
        self.input = input
        self.sigma = theano.shared(value=np.array(sigma, dtype=theano.config.floatX), name='sigma')
        apply_blur = T.gt(self.sigma, 0.0)
        no_blur = T.le(self.sigma, 0.0)
        self.output = ifelse(no_blur, input, gaussian_filter(input.dimshuffle('x', 0, 1), self.sigma, window_radius)[0, :, :])
        self.params = [self.sigma]


class Nonlinearity(object):
    def __init__(self, input, nonlinearity_ys = None):
        self.input = input
        #self.num_nonlinearity = num_nonlinearity
        if nonlinearity_ys is None:
            nonlinearity_ys = np.linspace(0, 1, num=20)
        nonlinearity_ys = nonlinearity_ys.astype(theano.config.floatX)
        self.nonlinearity_xs = theano.shared(value=np.linspace(0, 1, len(nonlinearity_ys)).astype(theano.config.floatX), name='nonlinearity_xs')
        self.nonlinearity_ys = theano.shared(value=nonlinearity_ys, name='nonlinearity_ys')
        self.output = nonlinearity(input, self.nonlinearity_xs, self.nonlinearity_ys, len(nonlinearity_ys))
        self.params = [self.nonlinearity_ys]


class LogNonlinearity(object):
    def __init__(self, input, nonlinearity_ys = None):
        self.input = input
        #self.num_nonlinearity = num_nonlinearity
        if nonlinearity_ys is None:
            nonlinearity_ys = np.linspace(0, 1, num=20)
        nonlinearity_ys = nonlinearity_ys.astype(theano.config.floatX)
        self.nonlinearity_xs = theano.shared(value=np.linspace(0, 1, len(nonlinearity_ys)).astype(theano.config.floatX), name='nonlinearity_xs')
        self.nonlinearity_ys = theano.shared(value=nonlinearity_ys, name='nonlinearity_ys')
        self.output = nonlinearity(input, self.nonlinearity_xs, T.exp(self.nonlinearity_ys), len(nonlinearity_ys))
        self.params = [self.nonlinearity_ys]


class CenterBias(object):
    def __init__(self, input, centerbias = None, alpha=1.0):
        self.input = input
        if centerbias is None:
            centerbias = np.ones(12)
        self.alpha = theano.shared(value = np.array(alpha).astype(theano.config.floatX), name='alpha')
        self.centerbias_ys = theano.shared(value=np.array(centerbias, dtype=theano.config.floatX), name='centerbias_ys')
        self.centerbias_xs = theano.shared(value=np.linspace(0, 1, len(centerbias), dtype=theano.config.floatX), name='centerbias_xs')

        height = T.cast(input.shape[0], theano.config.floatX)
        width = T.cast(input.shape[1], theano.config.floatX)
        x_coords = (T.arange(width) - 0.5*width) / (0.5*width)
        y_coords = (T.arange(height) - 0.5*height) / (0.5*height) + 0.0001  # We cannot have zeros in there because of grad

        x_coords = x_coords.dimshuffle('x', 0)
        y_coords = y_coords.dimshuffle(0, 'x')

        dists = T.sqrt(T.square(x_coords) + self.alpha*T.square(y_coords))
        self.max_dist = T.sqrt(1 + self.alpha)
        self.dists = dists/self.max_dist

        self.factors = nonlinearity(self.dists, self.centerbias_xs, self.centerbias_ys, len(centerbias))

        apply_centerbias = T.gt(self.centerbias_ys.shape[0], 2)
        self.output = ifelse(apply_centerbias, self.input*self.factors, self.input)
        self.params = [self.centerbias_ys, self.alpha]


class AdditiveCenterBias(object):
    def __init__(self, input, centerbias = None, alpha=1.0):
        self.input = input
        if centerbias is None:
            centerbias = np.ones(12)
        self.alpha = theano.shared(value = np.array(alpha).astype(theano.config.floatX), name='alpha')
        self.centerbias_ys = theano.shared(value=np.array(centerbias, dtype=theano.config.floatX), name='centerbias_ys')
        self.centerbias_xs = theano.shared(value=np.linspace(0, 1, len(centerbias), dtype=theano.config.floatX), name='centerbias_xs')

        height = T.cast(input.shape[0], theano.config.floatX)
        width = T.cast(input.shape[1], theano.config.floatX)
        x_coords = (T.arange(width) - 0.5*width) / (0.5*width)
        y_coords = (T.arange(height) - 0.5*height) / (0.5*height) + 0.0001  # We cannot have zeros in there because of grad

        x_coords = x_coords.dimshuffle('x', 0)
        y_coords = y_coords.dimshuffle(0, 'x')

        dists = T.sqrt(T.square(x_coords) + self.alpha*T.square(y_coords))
        self.max_dist = T.sqrt(1 + self.alpha)
        self.dists = dists/self.max_dist

        self.factors = nonlinearity(self.dists, self.centerbias_xs, self.centerbias_ys, len(centerbias))

        apply_centerbias = T.gt(self.centerbias_ys.shape[0], 2)
        self.output = ifelse(apply_centerbias, self.input+self.factors, self.input)
        self.params = [self.centerbias_ys, self.alpha]


class LogDensity(object):
    def __init__(self, input):
        self.input = input
        self.output = T.log(input / input.sum())


class LogDensityFromLogarithmicScale(object):
    def __init__(self, input):
        self.input = input
        self.output = input - T.log(T.exp(input).sum())


class AverageLogLikelihood(object):
    def __init__(self, log_densities, x_inds, y_inds):
        self.log_densities = log_densities
        self.log_likelihoods = log_densities[y_inds, x_inds]
        self.average_log_likelihood = self.log_likelihoods.mean()


class SaliencyMapProcessing(object):
    def __init__(self, saliency_map, x_inds = None, y_inds = None,
                 sigma = 0.0, window_radius = 80, nonlinearity_ys = None, centerbias = None, alpha = 1.0):

        self.saliency_map = saliency_map

        if x_inds is None:
            x_inds = T.lvector('x_inds')
        if y_inds is None:
            y_inds = T.lvector('y_inds')

        class TheanoObjects(object):
            pass

        self.theano_objects = TheanoObjects()

        self.x_inds = x_inds
        self.y_inds = y_inds
        self.theano_objects.blur = Blur(saliency_map, sigma=sigma, window_radius=window_radius)
        self.blur = self.theano_objects.blur.output

        self.theano_objects.nonlinearity = Nonlinearity(self.blur, nonlinearity_ys=nonlinearity_ys)
        self.nonlinearity = self.theano_objects.nonlinearity.output

        self.theano_objects.centerbias = CenterBias(self.nonlinearity, centerbias=centerbias, alpha=alpha)
        self.centerbias = self.theano_objects.centerbias.output

        self.theano_objects.log_density = LogDensity(self.centerbias)
        self.log_density = self.theano_objects.log_density.output

        self.theano_objects.average_log_likelihood = AverageLogLikelihood(self.log_density, self.x_inds, self.y_inds)

        self.average_log_likelihood = self.theano_objects.average_log_likelihood.average_log_likelihood
        self.params = self.theano_objects.blur.params + self.theano_objects.nonlinearity.params + self.theano_objects.centerbias.params

        self.blur_radius = self.theano_objects.blur.sigma
        self.nonlinearity_ys = self.theano_objects.nonlinearity.nonlinearity_ys
        self.centerbias_ys = self.theano_objects.centerbias.centerbias_ys
        self.alpha = self.theano_objects.centerbias.alpha


class SaliencyMapProcessingLogNonlinearity(object):
    def __init__(self, saliency_map, x_inds = None, y_inds = None,
                 sigma = 0.0, window_radius = 80, nonlinearity_ys = None, centerbias = None, alpha = 1.0):

        self.saliency_map = saliency_map

        if x_inds is None:
            x_inds = T.lvector('x_inds')
        if y_inds is None:
            y_inds = T.lvector('y_inds')

        class TheanoObjects(object):
            pass

        self.theano_objects = TheanoObjects()

        self.x_inds = x_inds
        self.y_inds = y_inds
        self.theano_objects.blur = Blur(saliency_map, sigma=sigma, window_radius=window_radius)
        self.blur = self.theano_objects.blur.output

        self.theano_objects.nonlinearity = LogNonlinearity(self.blur, nonlinearity_ys=nonlinearity_ys)
        self.nonlinearity = self.theano_objects.nonlinearity.output

        self.theano_objects.centerbias = CenterBias(self.nonlinearity, centerbias=centerbias, alpha=alpha)
        self.centerbias = self.theano_objects.centerbias.output

        self.theano_objects.log_density = LogDensity(self.centerbias)
        self.log_density = self.theano_objects.log_density.output

        self.theano_objects.average_log_likelihood = AverageLogLikelihood(self.log_density, self.x_inds, self.y_inds)

        self.average_log_likelihood = self.theano_objects.average_log_likelihood.average_log_likelihood
        self.params = self.theano_objects.blur.params + self.theano_objects.nonlinearity.params + self.theano_objects.centerbias.params

        self.blur_radius = self.theano_objects.blur.sigma
        self.nonlinearity_ys = self.theano_objects.nonlinearity.nonlinearity_ys
        self.centerbias_ys = self.theano_objects.centerbias.centerbias_ys
        self.alpha = self.theano_objects.centerbias.alpha


class SaliencyMapProcessingLogarithmic(object):
    def __init__(self, saliency_map, x_inds = None, y_inds = None,
                 sigma = 0.0, window_radius = 80, nonlinearity_ys = None, centerbias = None, alpha = 1.0):

        self.saliency_map = saliency_map

        if x_inds is None:
            x_inds = T.lvector('x_inds')
        if y_inds is None:
            y_inds = T.lvector('y_inds')

        class TheanoObjects(object):
            pass

        self.theano_objects = TheanoObjects()

        self.x_inds = x_inds
        self.y_inds = y_inds
        self.theano_objects.blur = Blur(saliency_map, sigma=sigma, window_radius=window_radius)
        self.blur = self.theano_objects.blur.output

        self.theano_objects.nonlinearity = Nonlinearity(self.blur, nonlinearity_ys=nonlinearity_ys)
        self.nonlinearity = self.theano_objects.nonlinearity.output

        self.theano_objects.centerbias = AdditiveCenterBias(self.nonlinearity, centerbias=centerbias, alpha=alpha)
        self.centerbias = self.theano_objects.centerbias.output

        self.theano_objects.log_density = LogDensityFromLogarithmicScale(self.centerbias)
        self.log_density = self.theano_objects.log_density.output

        self.theano_objects.average_log_likelihood = AverageLogLikelihood(self.log_density, self.x_inds, self.y_inds)

        self.average_log_likelihood = self.theano_objects.average_log_likelihood.average_log_likelihood
        self.params = self.theano_objects.blur.params + self.theano_objects.nonlinearity.params + self.theano_objects.centerbias.params

        self.blur_radius = self.theano_objects.blur.sigma
        self.nonlinearity_ys = self.theano_objects.nonlinearity.nonlinearity_ys
        self.centerbias_ys = self.theano_objects.centerbias.centerbias_ys
        self.alpha = self.theano_objects.centerbias.alpha
