from soccer_final.artifacts import verify


def test_committed_artifact_manifest() -> None:
    assert verify() == []
