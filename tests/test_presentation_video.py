import numpy as np

from soccer_final.presentation_video import live_probabilities, target_video_bitrate


def test_live_probabilities_are_normalized_and_directional() -> None:
    prematch = np.array([0.46, 0.21, 0.33])
    neutral = live_probabilities(prematch, 0.0)
    argentina = live_probabilities(prematch, 0.8)
    swiss = live_probabilities(prematch, -0.8)
    np.testing.assert_allclose(neutral.sum(), 1.0)
    assert argentina[0] > neutral[0]
    assert swiss[2] > neutral[2]


def test_target_bitrate_leaves_room_for_audio_and_container() -> None:
    bitrate = target_video_bitrate(duration=130.0, target_megabytes=50.0)
    assert 2800 < bitrate < 3200
