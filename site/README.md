# Match Intelligence 2026 — presentation site

The final-project presentation turns the paper, held-out evaluation, hardware profile, and YOLO match overlay into a single responsive story. The embedded finale is a 720p web edition of the verified 1080p presentation master; the original 50 MB and 200 MB delivery encodes remain in `../presentation/output/`.

## Run locally

Use Node 22 or newer and pnpm:

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`. Run `pnpm test` for a production build plus rendered-content and asset checks, and `pnpm lint` for static analysis.

## Presentation content

- Experiment scope: 3,961 matches, 24 competitions, 1958–2025
- Chronological split: 3,168 train / 793 held-out test
- Model: 36-feature regularized multinomial logistic regression
- Vision: YOLOv8m detections on 1,953 sampled frames
- Embedded video: `public/finale-web.mp4` (H.264/AAC, 1280×720)
- Paper download: `public/final-paper.pdf`

The generated social preview is `public/og.png`. Match frames and video are derived from the presentation input for local/classroom use; confirm the source-video rights before public redistribution.
