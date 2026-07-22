# Final presentation video

This extension turns the final research model into a narrated visual artifact:

1. obtain the attributed source clip locally;
2. run YOLOv8 player and sports-ball detection on Apple Silicon;
3. estimate team context from jersey-color evidence;
4. combine the final prematch model's probability vector with transparent,
   detection-derived momentum evidence;
5. render an annotated high-quality master with the original audio;
6. encode deliverables capped at 200 MB and 50 MB.

The overlay distinguishes the two signals:

- **Prematch engine** is the repository's leakage-safe logistic model. Since the
  training snapshot ends in 2025 and contains no 2026 competition-season state,
  the 2026 fixture uses neutral new-season state and is explicitly labeled as a
  prior rather than a fixture-specific historical forecast.
- **Live presentation odds** are a visualization, not a validated forecast. They
  update the prematch log probabilities with smoothed player territory, detected
  team counts, and ball proximity. They are included to explain the pipeline,
  not to claim causal win probability or support wagering.

## Rights and attribution

The local classroom presentation uses the public FIFA highlight clip titled
"Highlights | Argentina 3-1 Switzerland | FIFA World Cup 2026" from
<https://www.youtube.com/watch?v=zZxxDbLxEi4>. YouTube metadata declares no
downloadable license. The source MP4 is therefore excluded from Git, attribution
is burned into the overlay, and the derived files are intended for the final
course presentation. Confirm permission/fair-use requirements before any public
redistribution.

## Directories

```text
presentation/
├── source/       # local source MP4; gitignored
├── models/       # downloaded YOLO weights; gitignored
├── work/         # detections and intermediate renders; gitignored
└── output/       # final MP4 deliverables; Git LFS
```
