import pandas as pd

from soccer_final.features import FEATURE_COLUMNS, build_features


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "match_id": 1, "match_date": "2024-01-01", "competition_id": 1,
            "season_id": 1, "competition": "League", "season": "2024",
            "home_team_id": 10, "home_team": "A", "away_team_id": 20,
            "away_team": "B", "home_score": 2, "away_score": 0,
        },
        {
            "match_id": 2, "match_date": "2024-01-08", "competition_id": 1,
            "season_id": 1, "competition": "League", "season": "2024",
            "home_team_id": 20, "home_team": "B", "away_team_id": 10,
            "away_team": "A", "home_score": 1, "away_score": 1,
        },
    ])


def test_features_use_only_prior_results() -> None:
    result = build_features(sample_matches())
    assert not result[FEATURE_COLUMNS].isna().any().any()
    assert result.loc[0, "home_matches_played"] == 0
    assert result.loc[0, "away_matches_played"] == 0
    assert result.loc[1, "home_losses"] == 1
    assert result.loc[1, "away_wins"] == 1
    assert result.loc[1, "home_goals_against"] == 2
    assert result.loc[1, "away_goals_for"] == 2


def test_current_score_does_not_change_current_features() -> None:
    original = sample_matches()
    changed = original.copy()
    changed.loc[1, ["home_score", "away_score"]] = [9, 0]
    left = build_features(original).loc[1, FEATURE_COLUMNS]
    right = build_features(changed).loc[1, FEATURE_COLUMNS]
    pd.testing.assert_series_equal(left, right)
