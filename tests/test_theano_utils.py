from __future__ import absolute_import, print_function, division, unicode_literals

import unittest

import numpy as np
import theano
import theano.tensor as T

from pysaliency.theano_utils import nonlinearity


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
