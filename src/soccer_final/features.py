"""Leakage-safe prematch feature construction."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "home_matches_played",
    "away_matches_played",
    "home_wins",
    "away_wins",
    "home_draws",
    "away_draws",
    "home_losses",
    "away_losses",
    "home_goals_for",
    "away_goals_for",
    "home_goals_against",
    "away_goals_against",
    "home_points",
    "away_points",
    "home_form_points_5",
    "away_form_points_5",
    "home_avg_goals_for_5",
    "away_avg_goals_for_5",
    "home_avg_goals_against_5",
    "away_avg_goals_against_5",
    "home_goal_difference_5",
    "away_goal_difference_5",
    "home_matches_in_window_5",
    "away_matches_in_window_5",
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_team_home_matches",
    "home_team_home_win_rate",
    "home_team_home_goal_difference_per_match",
    "away_team_away_matches",
    "away_team_away_win_rate",
    "away_team_away_goal_difference_per_match",
    "home_rest_days",
    "away_rest_days",
    "rest_days_difference",
]


@dataclass
class TeamState:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    def update(self, goals_for: int, goals_against: int) -> None:
        self.matches += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        if goals_for > goals_against:
            self.wins += 1
            self.points += 3
        elif goals_for == goals_against:
            self.draws += 1
            self.points += 1
        else:
            self.losses += 1


@dataclass
class RollingState:
    points: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    goals_for: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    goals_against: deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def features(self) -> dict[str, float]:
        count = len(self.points)
        return {
            "form_points": float(sum(self.points)),
            "avg_goals_for": float(sum(self.goals_for) / count) if count else 0.0,
            "avg_goals_against": float(sum(self.goals_against) / count) if count else 0.0,
            "goal_difference": float(sum(self.goals_for) - sum(self.goals_against)),
            "matches_in_window": float(count),
        }

    def update(self, goals_for: int, goals_against: int) -> None:
        self.goals_for.append(goals_for)
        self.goals_against.append(goals_against)
        self.points.append(3 if goals_for > goals_against else 1 if goals_for == goals_against else 0)


def _target(home_score: int, away_score: int) -> int:
    return 0 if home_score > away_score else 2 if home_score < away_score else 1


def _venue_features(state: TeamState) -> tuple[float, float]:
    if not state.matches:
        return 0.0, 0.0
    return state.wins / state.matches, (state.goals_for - state.goals_against) / state.matches


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute features using only matches preceding each emitted row."""
    required = {
        "match_id", "match_date", "competition_id", "season_id", "home_team_id",
        "away_team_id", "home_score", "away_score", "competition", "season",
        "home_team", "away_team",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"matches are missing required columns: {missing}")
    ordered = matches.copy()
    ordered["match_date"] = pd.to_datetime(ordered["match_date"], errors="raise")
    ordered = ordered.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    overall: dict[tuple[int, int, int], TeamState] = {}
    rolling: dict[tuple[int, int, int], RollingState] = {}
    home_venue: dict[tuple[int, int, int], TeamState] = {}
    away_venue: dict[tuple[int, int, int], TeamState] = {}
    elo: dict[tuple[int, int, int], float] = {}
    last_date: dict[tuple[int, int, int], pd.Timestamp] = {}
    rows: list[dict[str, object]] = []

    for match in ordered.itertuples(index=False):
        home_key = (match.competition_id, match.season_id, match.home_team_id)
        away_key = (match.competition_id, match.season_id, match.away_team_id)
        for key in (home_key, away_key):
            overall.setdefault(key, TeamState())
            rolling.setdefault(key, RollingState())
            home_venue.setdefault(key, TeamState())
            away_venue.setdefault(key, TeamState())
            elo.setdefault(key, 1500.0)

        hs, aws = overall[home_key], overall[away_key]
        hr, ar = rolling[home_key].features(), rolling[away_key].features()
        hv, av = home_venue[home_key], away_venue[away_key]
        home_win_rate, home_venue_gd = _venue_features(hv)
        away_win_rate, away_venue_gd = _venue_features(av)
        home_rest = min((match.match_date - last_date[home_key]).days, 30) if home_key in last_date else 30
        away_rest = min((match.match_date - last_date[away_key]).days, 30) if away_key in last_date else 30
        home_elo, away_elo = elo[home_key], elo[away_key]

        rows.append({
            "match_id": match.match_id,
            "match_date": match.match_date,
            "competition": match.competition,
            "season": match.season,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "target": _target(match.home_score, match.away_score),
            "home_matches_played": hs.matches,
            "away_matches_played": aws.matches,
            "home_wins": hs.wins,
            "away_wins": aws.wins,
            "home_draws": hs.draws,
            "away_draws": aws.draws,
            "home_losses": hs.losses,
            "away_losses": aws.losses,
            "home_goals_for": hs.goals_for,
            "away_goals_for": aws.goals_for,
            "home_goals_against": hs.goals_against,
            "away_goals_against": aws.goals_against,
            "home_points": hs.points,
            "away_points": aws.points,
            "home_form_points_5": hr["form_points"],
            "away_form_points_5": ar["form_points"],
            "home_avg_goals_for_5": hr["avg_goals_for"],
            "away_avg_goals_for_5": ar["avg_goals_for"],
            "home_avg_goals_against_5": hr["avg_goals_against"],
            "away_avg_goals_against_5": ar["avg_goals_against"],
            "home_goal_difference_5": hr["goal_difference"],
            "away_goal_difference_5": ar["goal_difference"],
            "home_matches_in_window_5": hr["matches_in_window"],
            "away_matches_in_window_5": ar["matches_in_window"],
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
            "home_team_home_matches": hv.matches,
            "home_team_home_win_rate": home_win_rate,
            "home_team_home_goal_difference_per_match": home_venue_gd,
            "away_team_away_matches": av.matches,
            "away_team_away_win_rate": away_win_rate,
            "away_team_away_goal_difference_per_match": away_venue_gd,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_days_difference": home_rest - away_rest,
        })

        hs.update(match.home_score, match.away_score)
        aws.update(match.away_score, match.home_score)
        rolling[home_key].update(match.home_score, match.away_score)
        rolling[away_key].update(match.away_score, match.home_score)
        hv.update(match.home_score, match.away_score)
        av.update(match.away_score, match.home_score)
        last_date[home_key] = match.match_date
        last_date[away_key] = match.match_date
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
        actual_home = 1.0 if match.home_score > match.away_score else 0.0 if match.home_score < match.away_score else 0.5
        elo[home_key] = home_elo + 30.0 * (actual_home - expected_home)
        elo[away_key] = away_elo + 30.0 * ((1.0 - actual_home) - (1.0 - expected_home))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=Path("data/input/matches.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/prematch_features.parquet"))
    args = parser.parse_args()
    features = build_features(pd.read_parquet(args.matches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"wrote {len(features):,} rows and {len(FEATURE_COLUMNS)} features to {args.output}")


if __name__ == "__main__":
    main()
