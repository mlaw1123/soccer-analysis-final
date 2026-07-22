"""Data normalization and provenance helpers for StatsBomb match records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = {
    "match_id": "match_id",
    "match_date": "match_date",
    "competition.competition_id": "competition_id",
    "competition.competition_name": "competition",
    "season.season_id": "season_id",
    "season.season_name": "season",
    "home_team.home_team_id": "home_team_id",
    "home_team.home_team_name": "home_team",
    "away_team.away_team_id": "away_team_id",
    "away_team.away_team_name": "away_team",
    "home_score": "home_score",
    "away_score": "away_score",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_matches(source: Path) -> pd.DataFrame:
    """Return the minimal, analysis-ready columns from a StatsBomb match table."""
    frame = pd.read_parquet(source)
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"source table is missing required columns: {missing}")
    matches = frame[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="raise")
    integer_columns = [
        "match_id",
        "competition_id",
        "season_id",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    ]
    for column in integer_columns:
        matches[column] = pd.to_numeric(matches[column], errors="raise").astype("int64")
    if matches["match_id"].duplicated().any():
        raise ValueError("match_id must be unique")
    return matches.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def write_dataset(source: Path, output: Path, manifest_path: Path) -> None:
    matches = normalize_matches(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(output, index=False)
    class_counts = {
        "home_win": int((matches.home_score > matches.away_score).sum()),
        "draw": int((matches.home_score == matches.away_score).sum()),
        "away_win": int((matches.home_score < matches.away_score).sum()),
    }
    manifest = {
        "dataset": "StatsBomb Open Data match metadata",
        "upstream_source": "https://github.com/statsbomb/open-data",
        "upstream_snapshot": "Snapshot inherited from Soccer-Probability-Engine commit 14a61a6",
        "source_sha256": sha256_file(source),
        "normalized_path": str(output),
        "normalized_sha256": sha256_file(output),
        "rows": len(matches),
        "date_min": matches.match_date.min().date().isoformat(),
        "date_max": matches.match_date.max().date().isoformat(),
        "competitions": int(matches.competition.nunique()),
        "class_counts": class_counts,
        "columns": list(matches.columns),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/input/matches.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/input/dataset_manifest.json"))
    args = parser.parse_args()
    write_dataset(args.source, args.output, args.manifest)


if __name__ == "__main__":
    main()
