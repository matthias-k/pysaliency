from __future__ import absolute_import, print_function, division, unicode_literals

import theano
import theano.tensor as T


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
