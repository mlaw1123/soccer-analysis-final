# Data Card: StatsBomb Match Metadata

## Source

StatsBomb Open Data match metadata, normalized from the snapshot previously used by
`Soccer-Probability-Engine` at commit `14a61a6`. Upstream:
<https://github.com/statsbomb/open-data>.

## Contents

- 3,961 completed matches from 1958-06-24 through 2025-07-27
- 24 competitions
- Labels: 1,775 home wins; 879 draws; 1,307 away wins
- 3,168 chronological training rows and 793 chronological test rows
- 36 engineered features computed before updating state with the current result

## Lineage

- Feature table: `data/derived/prematch_features.parquet`
- Feature-table SHA-256: `1134f91e4b67f372c7e9edf39e464f4d3fa21c091e9c03750083c14c024df463`
- Normalized input and upstream-source hashes: `data/input/dataset_manifest.json`

## Known limitations

Coverage is curated and uneven across competitions, seasons, eras, and genders. The
study does not claim population-representative soccer performance. First matches in
each competition-season lack prior state; rest is capped at 30 days. Team state and
Elo reset at competition-season boundaries.
