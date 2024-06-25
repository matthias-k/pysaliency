import numpy as np

from pysaliency.metric_optimization_torch import maximize_expected_sim


def test_maximize_expected_sim_decay_1overk():
    density = np.ones((20, 20))
    density[6:17, 8:12] = 20
    density[2:4, 18:18] = 30
    density /= density.sum()
    log_density = np.log(density)

    saliency_map, score = maximize_expected_sim(
        log_density=log_density,
        kernel_size=1,
        train_samples_per_epoch=1000,
        val_samples=1000,
        max_iter=100
    )

    print(score)
    # We need a quite big tolerance in this test. Apparently there are
    # substantial differences between different systems, I'm not sure why.
    np.testing.assert_allclose(score, -0.8204902112483976, rtol=5e-4)


def test_maximize_expected_sim_decay_on_plateau():
    density = np.ones((20, 20))
    density[6:17, 8:12] = 20
    density[2:4, 18:18] = 30
    density /= density.sum()
    log_density = np.log(density)

    saliency_map, score = maximize_expected_sim(
        log_density=log_density,
        kernel_size=1,
        train_samples_per_epoch=1000,
        val_samples=1000,
        max_iter=100,
        backlook=1,
        min_iter=10,
        learning_rate_decay_scheme='validation_loss',
    )

    print(score)
    np.testing.assert_allclose(score, -0.8205618500709532, rtol=5e-4)  # need bigger tolerance to handle differences between CPU and GPU
