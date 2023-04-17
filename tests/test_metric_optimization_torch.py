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
    np.testing.assert_allclose(score, -0.8202784448862075, rtol=8e-7)  # need bigger tolerance to handle differences between CPU and GPU


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

    np.testing.assert_allclose(score, -0.8203513294458387, rtol=5e-7)  # need bigger tolerance to handle differences between CPU and GPU
