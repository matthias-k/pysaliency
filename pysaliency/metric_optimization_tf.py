from __future__ import print_function, division, absolute_import, unicode_literals

from tqdm import tqdm

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
import tensorflow as tf

from .models import sample_from_logdensity
from .tf_utils import gauss_blur


def sample_batch_fixations(log_density, fixations_per_image, batch_size, rst=None):
    xs, ys = sample_from_logdensity(log_density, fixations_per_image * batch_size, rst=rst)
    ns = np.repeat(np.arange(batch_size, dtype=int), repeats=fixations_per_image)

    return xs, ys, ns


def _eval_metric(log_density, test_samples, fn, seed=42, fixation_count=120, batch_size=50, verbose=True):
    values = []
    weights = []
    count = 0

    rst = np.random.RandomState(seed=seed)

    with tqdm(total=test_samples, leave=False, disable=not verbose) as t:
        while count < test_samples:
            this_count = min(batch_size, test_samples - count)
            xs, ys, ns = sample_batch_fixations(log_density, fixations_per_image=fixation_count, batch_size=this_count, rst=rst)

            values.append(fn(ns, ys, xs, this_count))
            weights.append(this_count)
            count += this_count
            t.update(this_count)
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    return np.average(values, weights=weights)


def constrained_descent(opt, loss, params, constraint, learning_rate):
    """
    opt: e.g. tf.train.GradientDescent()
    """
    assert len(params) == 1
    constraint_grad, = tf.gradients(constraint, params)

    constraint_grad_norm = tf.reduce_sum(tf.square(constraint_grad))
    normed_constraint_grad = constraint_grad / constraint_grad_norm

    grads_and_vars = opt.compute_gradients(loss, params)

    grads0 = grads_and_vars[0][0]
    params0 = params[0]
    # first step: make sure we are not running into negative values
    max_allowed_grad = params0 / learning_rate
    projected_grad1 = tf.reduce_min([grads0, max_allowed_grad], axis=0)

    # second step: Make sure that the gradient does not walk
    # out of the constraint
    projected_grad2 = projected_grad1 - tf.reduce_sum(projected_grad1 * constraint_grad) * normed_constraint_grad

    # projected_grad = grads0 - tf.reduce_sum(grads0*constraint_grad) * normed_constraint_grad

    # param = params[0]
    # active_set = (param <= 0)
    # active_grad = (projected_grad > 0)  # we are _descending_, i.e. grad>0 means we are trying to go down
    # mask = tf.logical_not(tf.logical_and(active_set, active_grad))
    # projected_grad = projected_grad * tf.cast(mask, 'float')

    grads_and_vars = [(projected_grad2, grads_and_vars[0][1])]

    tf_train_op = opt.apply_gradients(grads_and_vars)

    return tf_train_op


def build_fixation_maps(Ns, Ys, Xs, batch_size, height, width, dtype=tf.float32):
    indices = tf.stack((Ns, Ys, Xs), axis=1)

    fixation_maps = tf.scatter_nd(indices, updates=tf.ones((tf.shape(indices)[0], ), dtype=dtype),
                                  shape=(batch_size, height, width))

    return fixation_maps


def tf_similarity(saliency_map, empirical_saliency_maps):
    normalized_empirical_saliency_maps = empirical_saliency_maps / tf.reduce_sum(empirical_saliency_maps, reduction_indices=[1, 2], keepdims=True)
    normalized_saliency_map = saliency_map / tf.reduce_sum(saliency_map)
    minimums = tf.minimum(normalized_empirical_saliency_maps, tf.expand_dims(normalized_saliency_map, 0))

    similarities = tf.reduce_sum(minimums, reduction_indices=[1, 2])

    return similarities


def build_similarity_graph(saliency_map, ns, ys, xs, batch_size, height, width, kernel_size, truncate_gaussian, dtype=tf.float32):
    window_radius = int(kernel_size * truncate_gaussian)

    fixation_maps = build_fixation_maps(ns, ys, xs, batch_size, height, width, dtype=dtype)

    empirical_saliency_maps = gauss_blur(
        tf.expand_dims(fixation_maps, axis=3),
        kernel_size, windowradius=window_radius,
        mode='ZERO'
    )[:, :, :, 0]

    similarities = tf_similarity(saliency_map, empirical_saliency_maps)

    return similarities


def maximize_expected_sim(log_density, kernel_size,
                          train_samples_per_epoch, val_samples,
                          train_seed=43, val_seed=42,
                          fixation_count=100, batch_size=50,
                          max_batch_size=None,
                          verbose=True, session_config=None,
                          initial_learning_rate=1e-7,
                          backlook=1, min_iter=0, max_iter=1000,
                          truncate_gaussian=3,
                          learning_rate_decay_samples=None,
                          initial_saliency_map=None,
                          learning_rate_decay_scheme=None,
                          learning_rate_decay_ratio=0.333333333,
                          minimum_learning_rate=1e-11):
    """
       max_batch_size: maximum possible batch size to be used in validation
       learning rate decay samples: how often to decay the learning rate (using 1/k)

       learning_rate_decay_scheme: how to decay the learning rate:
           - None, "1/k": 1/k scheme
           - "validation_loss": if validation loss not better for last backlook
           steps

        learning_rate_decay_ratio: how much to decay learning rate if `learning_rate_decay_scheme` == 'validation_loss'
        minimum_learning_rate: stop optimization if learning rate would drop below this rate if using validation loss decay scheme

    """

    if max_batch_size is None:
        max_batch_size = batch_size

    if learning_rate_decay_scheme is None:
        learning_rate_decay_scheme = '1/k'

    if learning_rate_decay_samples is None:
        learning_rate_decay_samples = train_samples_per_epoch

    log_density_sum = logsumexp(log_density)
    if not -0.001 < log_density_sum < 0.001:
        raise ValueError("Log density not normalized! LogSumExp={}".format(log_density_sum))

    if initial_saliency_map is None:
        initial_value = gaussian_filter(np.exp(log_density), kernel_size, mode='constant')
    else:
        initial_value = initial_saliency_map

    if initial_value.min() < 0:
        initial_value -= initial_value.min()

    initial_value /= initial_value.sum()

    graph = tf.Graph()
    dtype = tf.float32

    height, width = log_density.shape

    with graph.as_default():
        # setting up
        saliency_map = tf.get_variable('saliency_map', shape=log_density.shape, dtype=dtype)

        Ns = tf.placeholder(tf.int32, shape=(None, ), name='ns')
        Ys = tf.placeholder(tf.int32, shape=(None, ), name='ys')
        Xs = tf.placeholder(tf.int32, shape=(None, ), name='xs')
        BatchSize = tf.placeholder(tf.int32, shape=(), name='batch_size')

        similarities = build_similarity_graph(saliency_map, Ns, Ys, Xs, BatchSize, height, width, kernel_size, truncate_gaussian, dtype=dtype)

        similarity = tf.reduce_mean(similarities)

        loss = -similarity

        # constraints

        constraint = tf.reduce_sum(saliency_map) - 1.0

        # training

        learning_rate = tf.Variable(1.0, dtype=dtype)
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        train_op = constrained_descent(opt, loss, [saliency_map], constraint, learning_rate)

        intermediate = saliency_map * tf.cast((saliency_map >= 0), 'float')
        normalized_saliency_map = intermediate / tf.reduce_sum(intermediate)
        normalize_op = tf.assign(saliency_map, normalized_saliency_map)

    # print("starting session")
    with tf.Session(graph=graph, config=session_config) as session:

        def _val_loss(ns, ys, xs, batch_size):
            # print("running", end='', flush=True)
            ret = session.run(loss, {Ns: ns, Ys: ys, Xs: xs, BatchSize: batch_size})
            # print("done")
            return ret

        def val_loss():
            return _eval_metric(log_density, val_samples, _val_loss, seed=val_seed,
                                fixation_count=fixation_count, batch_size=max_batch_size, verbose=False)

        session.run(tf.global_variables_initializer())
        # initial_value = gaussian_filter(density, kernel_size, mode='nearest')
        session.run(tf.assign(saliency_map, initial_value))
        session.run(tf.assign(learning_rate, initial_learning_rate))

        total_samples = 0
        decay_step = 0

        # print('starting val')

        val_scores = [val_loss()]
        learning_rate_relevant_scores = list(val_scores)
        train_rst = np.random.RandomState(seed=train_seed)
        # print('starting train')
        with tqdm(disable=not verbose) as outer_t:

            def general_termination_condition():
                return len(val_scores) - 1 >= max_iter

            def termination_1overk():
                return not (np.argmin(val_scores) >= len(val_scores) - backlook)

            def termination_validation():
                return session.run(learning_rate) < minimum_learning_rate

            def termination_condition():
                if len(val_scores) < min_iter:
                    return False
                cond = general_termination_condition()
                if learning_rate_decay_scheme == '1/k':
                    cond = cond or termination_1overk()
                elif learning_rate_decay_scheme == 'validation_loss':
                    cond = cond or termination_validation()

                return cond

            while not termination_condition():
                count = 0
                with tqdm(total=train_samples_per_epoch, leave=False, disable=True) as t:
                    while count < train_samples_per_epoch:
                        this_count = min(batch_size, train_samples_per_epoch - count)

                        xs, ys, ns = sample_batch_fixations(log_density, fixations_per_image=fixation_count, batch_size=this_count, rst=train_rst)
                        session.run(train_op, {Ns: ns, Ys: ys, Xs: xs, BatchSize: this_count})
                        session.run(normalize_op)

                        count += this_count
                        total_samples += this_count

                        if learning_rate_decay_scheme == '1/k':
                            if total_samples >= (decay_step + 1) * learning_rate_decay_samples:
                                decay_step += 1
                                session.run(tf.assign(learning_rate, initial_learning_rate / decay_step))

                        t.update(this_count)
                val_scores.append(val_loss())
                learning_rate_relevant_scores.append(val_scores[-1])

                if np.argmin(learning_rate_relevant_scores) < len(learning_rate_relevant_scores) - backlook:
                    old_learning_rate = session.run(learning_rate)
                    # print("Old Learning Rate", old_learning_rate, type(old_learning_rate))
                    # print("Decay", learning_rate_decay_ratio)
                    new_learning_rate = old_learning_rate * learning_rate_decay_ratio
                    session.run(tf.assign(learning_rate, new_learning_rate))
                    # print("Decaying learning_rate to", new_learning_rate)
                    learning_rate_relevant_scores = [learning_rate_relevant_scores[-1]]

                score1, score2 = val_scores[-2:]
                last_min = len(val_scores) - np.argmin(val_scores) - 1
                outer_t.set_description('{:.05f}, diff {:.02e}, best val {} steps ago, lr {:.02e}'.format(score2, score2 - score1, last_min, session.run(learning_rate)))
                outer_t.update(1)

        return session.run(saliency_map), val_scores[-1]
