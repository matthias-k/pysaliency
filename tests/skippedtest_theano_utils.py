from __future__ import absolute_import, print_function, division, unicode_literals

import unittest
import pytest

import numpy as np
from scipy.ndimage import gaussian_filter as scipy_filter
import theano
import theano.tensor as T

from pysaliency.theano_utils import nonlinearity, gaussian_filter, CenterBias, Blur


# mark whole module as theano
pytestmark = pytest.mark.theano


@pytest.fixture(params=['float64', 'float32'])
def dtype(request):
    return request.param


@pytest.fixture(params=['pixel', 'random'])
def input(request):
    return request.param


@pytest.fixture(params=[20.0])
def sigma(request):
    return request.param


class TestNonlinearity(unittest.TestCase):
    def setUp(self):
        self.x = T.dvector('x')
        self.x.tag.test_value = np.linspace(0, 1, 20)
        self.y = T.dvector('y')
        self.y.tag.test_value = np.linspace(0, 1, 20)
        self.input = T.dvector('input')
        self.input.tag.test_value = np.linspace(0, 1, 20)
        self.length = 20
        self.nonlin = nonlinearity(self.input, self.x, self.y, self.length)
        self.f = theano.function([self.input, self.x, self.y], self.nonlin)

    def test_id(self):
        x = np.linspace(0, 1, self.length)
        y = np.linspace(0, 1, self.length)
        inp = np.linspace(0, 1, self.length)
        out = self.f(inp, x, y)
        np.testing.assert_allclose(out, inp)

    def test_mult_id(self):
        x = np.linspace(0, 1, self.length)
        y = np.linspace(0, 2, self.length)
        inp = np.linspace(0, 1, self.length)
        out = self.f(inp, x, y)
        np.testing.assert_allclose(out, y)

    def test_shifted_id(self):
        x = np.linspace(0, 1, self.length)
        y = np.linspace(0, 1, self.length)+1
        inp = np.linspace(0, 1, self.length)
        out = self.f(inp, x, y)
        np.testing.assert_allclose(out, y)

    def test_random(self):
        x = np.linspace(0, 1, self.length)
        y = np.random.randn(self.length)
        inp = np.linspace(0, 1, self.length)
        out = self.f(inp, x, y)
        np.testing.assert_allclose(out, y)

    def test_constant(self):
        x = np.linspace(0, 1, self.length)
        y = np.ones(self.length)
        inp = np.linspace(0, 1, self.length)
        out = self.f(inp, x, y)
        np.testing.assert_allclose(out, y)


class TestBlur(object):
    def setUp(self):
        theano.config.compute_test_value = 'ignore'

    def test_blur_zeros(self):
        sigma = theano.shared(20.0)
        window_radius = 20*4
        data = T.tensor3('data', dtype='float64')
        data.tag.test_value = np.zeros((1, 10, 10))

        blur = gaussian_filter(data, sigma, window_radius)

        f = theano.function([data], blur)

        test_data = np.zeros((1000, 1000))
        out = f(test_data[np.newaxis, :, :])[0, :, :]

        np.testing.assert_allclose(out, 0)

    def test_blur_ones(self):
        sigma = theano.shared(20.0)
        window_radius = 20*4
        data = T.tensor3('data', dtype='float64')
        data.tag.test_value = np.zeros((1, 10, 10))

        blur = gaussian_filter(data, sigma, window_radius)

        f = theano.function([data], blur)

        test_data = np.ones((1000, 1000))
        out = f(test_data[np.newaxis, :, :])[0, :, :]

        np.testing.assert_allclose(out, 1)

    @pytest.mark.skip("Doesn't seem to work with theano right now")
    def test_other(self, dtype, input, sigma):
        theano.config.compute_test_value = 'ignore'
        sigma_theano = theano.shared(sigma)
        window_radius = int(sigma*4)
        if input == 'pixel':
            test_data = np.ones((100, 100))
            test_data[50, 50] = 2
        elif input == 'random':
            test_data = 10*np.ones((100, 100))
            test_data += np.random.randn(100, 100)

        else:
            raise ValueError(input)

        test_data = test_data.astype(dtype)
        data = T.tensor3('data', dtype=dtype)
        data.tag.test_value = test_data[np.newaxis, :, :]
        print(data.dtype)

        blur = gaussian_filter(data, sigma_theano, window_radius)

        f = theano.function([data], blur)
        out = f(test_data[np.newaxis, :, :])[0, :, :]

        scipy_out = scipy_filter(test_data, sigma, mode='nearest')

        if dtype == 'float32':
            rtol = 5e-6
        else:
            rtol = 1e-7
        np.testing.assert_allclose(out, scipy_out, rtol=rtol)


class TestBlurObject(object):
    def test_blur(self):
        theano.config.floatX = 'float64'
        data = T.matrix('data')
        data.tag.test_value = np.random.randn(10, 10)
        blur = Blur(data, sigma=20.0, window_radius = 80)

        tmp = np.random.randn(1000, 2000)
        tmp += 10.0
        out = blur.output.eval({data: tmp})
        scipy_out = scipy_filter(tmp, 20.0, mode='nearest')
        np.testing.assert_allclose(out, scipy_out)

    def test_no_blur(self):
        theano.config.floatX = 'float64'
        data = T.matrix('data')
        data.tag.test_value = np.random.randn(10, 10)
        blur = Blur(data, sigma=0.0, window_radius = 80)

        tmp = np.random.randn(1000, 2000)
        tmp += 10.0
        out = blur.output.eval({data: tmp})
#        scipy_out = scipy_filter(tmp, 20.0, mode='nearest')
        np.testing.assert_allclose(out, tmp)


class TestCenterBias(object):
    def test_centerbias_ones(self):
        theano.config.floatX = 'float64'
        data = T.matrix('data')
        data.tag.test_value = np.random.randn(10, 10)
        center_bias = CenterBias(data)

        tmp = np.ones((1000, 2000))
        out = center_bias.output.eval({data: tmp})
        np.testing.assert_allclose(out, tmp)

    def test_centerbias_ones_times_two(self):
        theano.config.floatX = 'float64'
        data = T.matrix('data')
        data.tag.test_value = np.random.randn(10, 10)
        center_bias = CenterBias(data, centerbias=np.array([2.0, 2.0, 2.0]))

        tmp = np.ones((1000, 2000))
        out = center_bias.output.eval({data: tmp})
        np.testing.assert_allclose(out, 2*tmp)

    def test_centerbias_random(self):
        theano.config.floatX = 'float64'
        data = T.matrix('data')
        data.tag.test_value = np.random.randn(10, 10)
        center_bias = CenterBias(data)

        tmp = np.random.randn(1000, 2000)
        out = center_bias.output.eval({data: tmp})
        np.testing.assert_allclose(out, tmp)

    def test_centerbias_ones_nontrivial(self):
        theano.config.floatX = 'float64'
        data = T.matrix('data')
        data.tag.test_value = np.random.randn(10, 10)
        center_bias = CenterBias(data, centerbias=np.array([0.0, 0.0, 1.0, 1.0]))

        tmp = np.ones((1000, 2000))
        out = center_bias.output.eval({data: tmp})
        np.testing.assert_allclose(out.min(), 0.0)
        np.testing.assert_allclose(out.max(), 1.0)

    def test_centerbias_empty(self):
        theano.config.floatX = 'float64'
        data = T.matrix('data')
        data.tag.test_value = np.random.randn(10, 10)
        center_bias = CenterBias(data, centerbias = np.array([1.0]), alpha=3.0)

        tmp = np.random.randn(1000, 2000)
        out = center_bias.output.eval({data: tmp})
        np.testing.assert_allclose(out, tmp)
