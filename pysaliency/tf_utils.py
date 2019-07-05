from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
slim = tf.contrib.slim


def normalize_axis(input_tensor, axis):
    """ convert negative indices into positive indices since tensorflow can't handle them """
    if axis < 0:
        ndims = len(input_tensor.get_shape())
        axis = ndims + axis
    return axis


def replication_padding(input_tensor, axis=0, size=1):
    """ add replication padding to a tensor along a given axis """
    with tf.name_scope('replication_padding'):
        if not isinstance(size, (tuple, list)):
            size = (size, size)
        ndims = len(input_tensor.get_shape())
        axis = normalize_axis(input_tensor, axis)
        start_slice_obj = [slice(None)] * axis + [slice(0, 1)]
        start_slice = input_tensor[start_slice_obj]
        repeats = [1] * axis + [size[0]] + [1] * (ndims-axis-1)
        start_part = tf.tile(start_slice, repeats)
        end_slice_obj = [slice(None)] * axis + [slice(-1, None)]
        end_slice = input_tensor[end_slice_obj]
        repeats = [1] * axis + [size[1]] + [1] * (ndims-axis-1)
        end_part = tf.tile(end_slice, repeats)
        return tf.concat((start_part, input_tensor, end_part), axis=axis)


def get_gaussian_kernel(sigma, windowradius=5):
    with tf.name_scope('gaussian_kernel'):
        kernel = tf.cast(tf.range(0, 2*windowradius+1), 'float') - windowradius
        kernel = tf.exp(-(kernel**2)/(2*sigma**2))
        kernel /= tf.reduce_sum(kernel)
        return kernel


def blowup_1d_kernel(kernel, axis=-1):
    #with tf.name_scope("blowup_1d_kernel")
    assert isinstance(axis, int)

    shape = [1 for i in range(4)]
    shape[axis] = -1
    return tf.reshape(kernel, shape)


@slim.add_arg_scope
def gaussian_convolution_along_axis(inputs, axis, sigma, windowradius=5, mode='NEAREST', scope=None,
                                    outputs_collections=None):
    with tf.name_scope(scope, 'gauss_1d', [inputs, sigma, windowradius]):
        if mode == 'NEAREST':
            inputs = replication_padding(inputs, axis=axis+1, size=windowradius)
        elif mode == 'ZERO':
            paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]
            paddings[axis+1] = [windowradius, windowradius]
            inputs = tf.pad(inputs, paddings)
        elif mode == 'VALID':
            pass
        else:
            raise ValueError(mode)

        kernel_1d = get_gaussian_kernel(sigma, windowradius=windowradius)
        kernel = blowup_1d_kernel(kernel_1d, axis)
        #print(windowradius)

        output = tf.nn.conv2d(inputs, kernel,
                              strides=[1, 1, 1, 1], padding="VALID", name='gaussian_convolution')
        return output

        #return slim.utils.collect_named_outputs(outputs_collections, sc, output)


@slim.add_arg_scope
def gauss_blur(inputs, sigma, windowradius=5, mode='NEAREST', scope=None,
               outputs_collections=None):
    with tf.name_scope(scope, 'gauss_blur', [inputs, sigma, windowradius]) as sc:

        outputs = inputs

        for axis in [0, 1]:

            outputs = gaussian_convolution_along_axis(outputs,
                                                      axis=axis,
                                                      sigma=sigma,
                                                      windowradius=windowradius,
                                                      mode=mode)
        return outputs

        return slim.utils.collect_named_outputs(outputs_collections, sc, outputs)


def tf_logsumexp(data, axis=0):
    """computes logsumexp along axis in its own graph and session"""
    with tf.Graph().as_default() as g:
        input_tensor = tf.placeholder(tf.float32, name='input_tensor')
        output_tensor = tf.reduce_logsumexp(input_tensor, axis=axis)

        with tf.Session(graph=g) as sess:
            return sess.run(output_tensor, {input_tensor: data})
