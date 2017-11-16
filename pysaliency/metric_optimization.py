from __future__ import print_function, division, absolute_import, unicode_literals

from tqdm import tqdm

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.misc import logsumexp
import tensorflow as tf

from .models import sample_from_logdensity
from .saliency_map_models import SaliencyMapModel
from .tf_utils import gauss_blur


def sample_batch_fixations(log_density, fixations_per_image, batch_size, rst=None):
    xs, ys = sample_from_logdensity(log_density, fixations_per_image*batch_size, rst=rst)
    ns = np.repeat(np.arange(batch_size, dtype=int), repeats=fixations_per_image)

    return xs, ys, ns


def _eval_metric(log_density, test_samples, fn, seed=42, fixation_count=120, batch_size=50, verbose=True):
    values = []
    weights = []
    count = 0

    rst = np.random.RandomState(seed=seed)

    with tqdm(total=test_samples, leave=False, disable=not verbose) as t:
        while count < test_samples:
            this_count = min(batch_size, test_samples-count)
            xs, ys, ns = sample_batch_fixations(log_density, fixations_per_image=fixation_count, batch_size=this_count, rst=rst)

            values.append(fn(ns, ys, xs, this_count))
            weights.append(this_count)
            count += this_count
            t.update(this_count)
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    return np.average(values, weights=weights)


def constrained_descent(opt, loss, params, constraint):
    """
    opt: e.g. tf.train.GradientDescent()
    """
    assert len(params) == 1
    constraint_grad, = tf.gradients(constraint, params)

    constraint_grad_norm = tf.reduce_sum(tf.square(constraint_grad))
    normed_constraint_grad = constraint_grad / constraint_grad_norm

    grads_and_vars = opt.compute_gradients(loss, params)

    grads0 = grads_and_vars[0][0]
    projected_grad = grads0 - tf.reduce_sum(grads0*constraint_grad) * normed_constraint_grad

    grads_and_vars = [(projected_grad, grads_and_vars[0][1])]

    tf_train_op = opt.apply_gradients(grads_and_vars)

    return tf_train_op


def build_fixation_maps(Ns, Ys, Xs, batch_size, height, width, dtype=tf.float32):
    indices = tf.stack((Ns, Ys, Xs), axis=1)

    fixation_maps = tf.scatter_nd(indices, updates=tf.ones((tf.shape(indices)[0], ), dtype=dtype),
                                  shape=(batch_size, height, width))

    return fixation_maps


def tf_similarity(saliency_map, empirical_saliency_maps):
    normalized_empirical_saliency_maps = empirical_saliency_maps / tf.reduce_sum(empirical_saliency_maps, reduction_indices=[1, 2], keep_dims=True)
    normalized_saliency_map = saliency_map / tf.reduce_sum(saliency_map)
    minimums = tf.minimum(normalized_empirical_saliency_maps, tf.expand_dims(normalized_saliency_map, 0))

    similarities = tf.reduce_sum(minimums, reduction_indices=[1, 2])

    return similarities


def build_similarity_graph(saliency_map, ns, ys, xs, batch_size, height, width, kernel_size, truncate_gaussian, dtype=tf.float32):
    window_radius = int(kernel_size*truncate_gaussian)

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
                          learning_rate_decay_samples=None):
    """
       max_batch_size: maximum possible batch size to be used in validation
       learning rate decay samples: how often to decay the learning rate (using 1/k)
    """

    if max_batch_size is None:
        max_batch_size = batch_size

    if learning_rate_decay_samples is None:
        learning_rate_decay_samples = train_samples_per_epoch

    log_density_sum = logsumexp(log_density)
    if not -0.001 < log_density_sum < 0.001:
        raise ValueError("Log density not normalized! LogSumExp={}".format(log_density_sum))

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

        train_op = constrained_descent(opt, loss, [saliency_map], constraint)

    #print("starting session")
    with tf.Session(graph=graph, config=session_config) as session:

        def _val_loss(ns, ys, xs, batch_size):
            #print("running", end='', flush=True)
            ret = session.run(loss, {Ns: ns, Ys: ys, Xs: xs, BatchSize: batch_size})
            #print("done")
            return ret

        def val_loss():
            return _eval_metric(log_density, val_samples, _val_loss, seed=val_seed,
                                fixation_count=fixation_count, batch_size=max_batch_size, verbose=False)

        session.run(tf.global_variables_initializer())
        #initial_value = gaussian_filter(density, kernel_size, mode='nearest')
        initial_value = gaussian_filter(np.exp(log_density), kernel_size, mode='constant')

        initial_value /= initial_value.sum()
        session.run(tf.assign(saliency_map, initial_value))
        session.run(tf.assign(learning_rate, initial_learning_rate))

        total_samples = 0
        decay_step = 0

        #print('starting val')

        val_scores = [val_loss()]
        train_rst = np.random.RandomState(seed=train_seed)
        #print('starting train')
        with tqdm(disable=not verbose) as outer_t:
            while (len(val_scores) - 1 < max_iter) and (len(val_scores) < min_iter or np.argmin(val_scores) >= len(val_scores) - backlook):
                count = 0
                with tqdm(total=train_samples_per_epoch, leave=False, disable=True) as t:
                    while count < train_samples_per_epoch:
                        this_count = min(batch_size, train_samples_per_epoch-count)

                        xs, ys, ns = sample_batch_fixations(log_density, fixations_per_image=fixation_count, batch_size=this_count, rst=train_rst)
                        session.run(train_op, {Ns: ns, Ys: ys, Xs: xs, BatchSize: this_count})

                        count += this_count
                        total_samples += this_count

                        if total_samples >= (decay_step+1)*learning_rate_decay_samples:
                            decay_step += 1
                            session.run(tf.assign(learning_rate, initial_learning_rate/decay_step))

                        t.update(this_count)
                val_scores.append(val_loss())

                score1, score2 = val_scores[-2:]
                last_min = len(val_scores) - np.argmin(val_scores) - 1
                outer_t.set_description('{:.05f}, diff {:.02e} [{}]'.format(score2, score2-score1, last_min))
                outer_t.update(1)

        return session.run(saliency_map), val_scores[-1]


class SIMSaliencyMapModel(SaliencyMapModel):
    def __init__(self, parent_model,
                 kernel_size,
                 train_samples_per_epoch=1000, val_samples=1000,
                 train_seed=43, val_seed=42,
                 fixation_count=100, batch_size=50,
                 max_batch_size=None,
                 initial_learning_rate=1e-7,
                 backlook=1,
                 min_iter=0,
                 max_iter=1000,
                 truncate_gaussian=3,
                 learning_rate_decay_samples=None,
                 verbose=True,
                 session_config=None,
                 **kwargs
                 ):
        super(SIMSaliencyMapModel, self).__init__(**kwargs)
        self.parent_model = parent_model

        self.kernel_size = kernel_size
        self.train_samples_per_epoch = train_samples_per_epoch
        self.val_samples = val_samples
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.fixation_count = fixation_count
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.initial_learning_rate = initial_learning_rate
        self.backlook = backlook
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.truncate_gaussian = truncate_gaussian
        self.learning_rate_decay_samples = learning_rate_decay_samples
        self.verbose = verbose
        self.session_config = session_config

    def _saliency_map(self, stimulus):
        log_density = self.parent_model.log_density(stimulus)
        saliency_map, val_scores = maximize_expected_sim(
            log_density,
            kernel_size=self.kernel_size,
            train_samples_per_epoch=self.train_samples_per_epoch,
            val_samples=self.val_samples,
            train_seed=self.train_seed,
            val_seed=self.val_seed,
            fixation_count=self.fixation_count,
            batch_size=self.batch_size,
            max_batch_size=self.max_batch_size,
            verbose=self.verbose,
            session_config=self.session_config,
            initial_learning_rate=self.initial_learning_rate,
            backlook=self.backlook,
            min_iter=self.min_iter,
            max_iter=self.max_iter,
            truncate_gaussian=self.truncate_gaussian,
            learning_rate_decay_samples=self.learning_rate_decay_samples,
        )
        return saliency_map
