# Local hardware and processing constraints

Recorded on 22 July 2026:

- MacBook Pro, model `MacBookPro18,1`
- Apple M1 Pro
- 10 CPU cores: 8 performance and 2 efficiency
- 16 GB unified memory
- arm64 macOS 15.7.3 / Darwin 24.6.0
- no discrete CUDA GPU

## Processing choices

- Use PyTorch MPS when available, with CPU fallback.
- Use a single YOLOv8 model instance and stream frames rather than buffering the
  clip in memory.
- Cache compact detections so overlay and compression iterations do not repeat
  inference.
- Preserve the 1080p source for the high-quality master.
- Use Apple VideoToolbox H.264 for the master when reliable; use deterministic
  two-pass `libx264` bitrate control for the strict 200 MB and 50 MB caps.
- Keep working data and downloaded weights out of Git. Track final MP4s through
  Git LFS because the 200 MB deliverable exceeds GitHub's normal file limit.

The source clip is about 3,900 frames. YOLO inference is the dominant workload;
compression is substantially cheaper and can be rerun from the annotated master.

## Benchmark

YOLOv8m at a 960-pixel inference size on the MPS device processed 60 sampled
1080p frames in 11.31 seconds during cold-start benchmarking. Once warmed, the
full 1,953-frame inference pass completed in 146.37 seconds, sustaining 13.34
sampled frames per second. The lossless overlay rendered all 3,906 source frames
in 90.59 seconds (43.1 fps overall) and produced a 2.15 GiB master.
