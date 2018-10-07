"""
Author: Matthias Kuemmerer, 2014

"""
from __future__ import print_function, division, unicode_literals, absolute_import

import sys
import numpy as np


class FunctionWithApproxJacobian(object):
    def __init__(self, func, epsilon, verbose=True):
        self._func = func
        self.epsilon = epsilon
        self.value_cache = {}
        self.verbose = verbose

    def __call__(self, x, *args, **kwargs):
        key = tuple(x)
        if not key in self.value_cache:
            self.log('.')
            value = self._func(x, *args, **kwargs)
            if np.any(np.isnan(value)):
                print("Warning! nan function value encountered at {0}".format(x))
            self.value_cache[key] = value
        return self.value_cache[key]

    def func(self, x, *args, **kwargs):
        if self.verbose:
            print(x)
        return self(x, *args, **kwargs)

    def log(self, msg):
        if self.verbose:
            sys.stdout.write(msg)
            sys.stdout.flush()

    def jac(self, x, *args, **kwargs):
        self.log('G[')
        x0 = np.asfarray(x)
        #print x0
        dxs = np.zeros((len(x0), len(x0) + 1))
        for i in range(len(x0)):
            dxs[i, i + 1] = self.epsilon
        results = [self(*(x0 + dxs[:, i], ) + args, **kwargs) for i in range(len(x0) + 1)]
        jac = np.zeros([len(x0), len(np.atleast_1d(results[0]))])
        for i in range(len(x0)):
            jac[i] = (results[i + 1] - results[0]) / self.epsilon
        self.log(']')
        return jac.transpose()


class FunctionWithApproxJacobianCentral(FunctionWithApproxJacobian):
    def jac(self, x, *args, **kwargs):
        self.log('G[')
        x0 = np.asfarray(x)
        #print x0
        dxs = np.zeros((len(x0), 2*len(x0)))
        for i in range(len(x0)):
            dxs[i, i] = -self.epsilon
            dxs[i, len(x0)+i] = self.epsilon
        results = [self(*(x0 + dxs[:, i], ) + args, **kwargs) for i in range(2*len(x0))]
        jac = np.zeros([len(x0), len(np.atleast_1d(results[0]))])
        for i in range(len(x0)):
            jac[i] = (results[len(x0)+i] - results[i]) / (2*self.epsilon)
        self.log(']')
        return jac.transpose()
