import numpy as np

from soccer_final.experiment import calibration_bins, metric_row, temperature_scale


def test_temperature_scaling_preserves_probability_rows() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    scaled = temperature_scale(probabilities, 1.5)
    np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
    np.testing.assert_array_equal(scaled.argmax(axis=1), probabilities.argmax(axis=1))


def test_metrics_and_calibration_bins() -> None:
    labels = np.array([0, 1, 2])
    probabilities = np.eye(3) * 0.8 + (1 - np.eye(3)) * 0.1
    metrics = metric_row(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] > 0
    bins = calibration_bins(labels, probabilities)
    assert bins["count"].sum() == 3
