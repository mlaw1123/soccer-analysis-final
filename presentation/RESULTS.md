# Presentation video results

## Source and inference

- Source: FIFA YouTube highlight `zZxxDbLxEi4`
- Duration: 130.33 seconds
- Resolution: 1920x1080 at 29.97 fps
- Source frames: 3,906
- YOLOv8m inference frames: 1,953 (stride 2)
- Device: PyTorch MPS on Apple M1 Pro
- Inference size: 960 pixels
- Full inference time: 146.37 seconds
- Sustained inference rate: 13.34 sampled frames/second

YOLO produced 25,090 retained person detections and 1,865 sports-ball
detections. The ball was detected in 1,163 sampled frames (59.55%). Jersey-color
classification assigned 10,007 person detections to Argentina, 10,726 to
Switzerland, and conservatively left 4,357 as other/person. The nearest-player
context proxy resolved 533 sampled frames to Argentina, 450 to Switzerland, and
970 to unclear.

## Probability layers

The final prematch logistic model was evaluated at neutral new-season state
because its versioned dataset ends in 2025 and cannot provide 2026 competition
history:

- Argentina: 46.59%
- Draw: 20.28%
- Switzerland: 33.14%

The changing on-screen values are explicitly labeled research visualization
odds. They combine those fixed log probabilities with bounded, four-second
smoothed detection context. They are not empirically calibrated live-match odds.

## Encodes

| Artifact | Codec | Resolution | Size | SHA-256 |
| --- | --- | --- | ---: | --- |
| Lossless annotated master | H.264 lossless 4:4:4 + AAC | 1920x1080 | 2,310,134,127 bytes | `e2a5eeb7f6a6597ac35f7ec23ddb2fd6bc7d63f324f29c826f5d96d230d80325` |
| 200 MB delivery | H.264 4:2:0 + AAC | 1920x1080 | 187,446,375 bytes (178.76 MiB) | `6a9d6e106d085899b2ef8597c53ba4aebaaa376dc26d04b77e496566891bf198` |
| 50 MB delivery | H.264 4:2:0 + AAC | 1920x1080 | 46,338,081 bytes (44.19 MiB) | `3f47a37cab5883051796ec09f9d7afc7febcabb595e9e77d5404f74e83e7a8e6` |

Both delivery MP4s decode end to end with no FFmpeg errors and contain H.264
video plus AAC audio. Representative frames at six timestamps were inspected
from the master. A three-way master/200 MB/50 MB comparison at 55 seconds showed
readable probability text and detection labels in all variants; the expected
detail loss is visible in the 50 MB version but does not compromise presentation
legibility.

The 2.15 GiB master and downloaded source are reproducible local working files
under `presentation/work/` and `presentation/source/` and are intentionally
gitignored. The two delivery MP4s are tracked with Git LFS.
