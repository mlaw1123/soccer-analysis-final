"""Create the final presentation video with detections and probability overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd

from soccer_final.experiment import align_probabilities, temperature_scale
from soccer_final.features import FEATURE_COLUMNS


LABELS = ("Argentina", "Draw", "Switzerland")
ARGENTINA = "ARG"
SWITZERLAND = "SUI"
NEUTRAL = "OTHER"
COLORS = {
    ARGENTINA: (235, 190, 70),
    SWITZERLAND: (40, 45, 225),
    NEUTRAL: (180, 180, 180),
    "BALL": (40, 235, 250),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prematch_probabilities(bundle_path: Path) -> np.ndarray:
    """Evaluate the final prematch model at transparent neutral 2026 state."""
    bundle = joblib.load(bundle_path)
    values = {column: 0.0 for column in FEATURE_COLUMNS}
    values.update(
        home_elo=1500.0,
        away_elo=1500.0,
        home_rest_days=30.0,
        away_rest_days=30.0,
    )
    raw = align_probabilities(bundle["model"], pd.DataFrame([values]))
    return temperature_scale(raw, float(bundle["temperature"]))[0]


def classify_jersey(frame: np.ndarray, box: list[float]) -> tuple[str, dict[str, float]]:
    """Use transparent color evidence to separate Argentina, Switzerland, and others."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    box_height = max(1, y2 - y1)
    box_width = max(1, x2 - x1)
    torso = frame[
        y1 + int(box_height * 0.18) : y1 + int(box_height * 0.58),
        x1 + int(box_width * 0.12) : x2 - int(box_width * 0.12),
    ]
    if torso.size == 0:
        return NEUTRAL, {"argentina": 0.0, "switzerland": 0.0}
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue, saturation, value = (hsv[..., index] for index in range(3))
    valid = value > 45
    denominator = max(1, int(valid.sum()))
    red = (((hue <= 12) | (hue >= 168)) & (saturation >= 90) & valid).sum() / denominator
    white = ((saturation <= 72) & (value >= 125) & valid).sum() / denominator
    sky_blue = (((hue >= 82) & (hue <= 112)) & (saturation >= 35) & (value >= 90) & valid).sum() / denominator
    argentina_score = float(white + 0.7 * sky_blue)
    switzerland_score = float(red)
    if switzerland_score >= 0.16 and switzerland_score > argentina_score * 0.75:
        team = SWITZERLAND
    elif argentina_score >= 0.24:
        team = ARGENTINA
    else:
        team = NEUTRAL
    return team, {"argentina": argentina_score, "switzerland": switzerland_score}


def is_pitch_person(box: list[float], frame_shape: tuple[int, ...]) -> bool:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    box_height, box_width = y2 - y1, x2 - x1
    return (
        box_height >= height * 0.025
        and box_height >= box_width * 0.85
        and y2 >= height * 0.20
        and x2 >= 0
        and x1 <= width
    )


def frame_context(detections: list[dict[str, Any]], frame_shape: tuple[int, ...]) -> dict[str, Any]:
    height, width = frame_shape[:2]
    diagonal = math.hypot(width, height)
    players = [item for item in detections if item["class_id"] == 0 and item["team"] in (ARGENTINA, SWITZERLAND)]
    balls = [item for item in detections if item["class_id"] == 32]
    counts = {ARGENTINA: 0, SWITZERLAND: 0}
    confidence = {ARGENTINA: 0.0, SWITZERLAND: 0.0}
    for player in players:
        counts[player["team"]] += 1
        confidence[player["team"]] += float(player["confidence"])
    evidence = (confidence[ARGENTINA] - confidence[SWITZERLAND]) / max(
        1.0, confidence[ARGENTINA] + confidence[SWITZERLAND]
    )
    possession = "unclear"
    if balls and players:
        ball = max(balls, key=lambda item: item["confidence"])
        bx = (ball["box"][0] + ball["box"][2]) / 2
        by = (ball["box"][1] + ball["box"][3]) / 2
        nearest = min(
            players,
            key=lambda item: math.hypot(
                (item["box"][0] + item["box"][2]) / 2 - bx,
                (item["box"][1] + item["box"][3]) / 2 - by,
            ),
        )
        distance = math.hypot(
            (nearest["box"][0] + nearest["box"][2]) / 2 - bx,
            (nearest["box"][1] + nearest["box"][3]) / 2 - by,
        ) / diagonal
        if distance < 0.18:
            possession = nearest["team"]
            proximity = max(0.0, 1.0 - distance / 0.18)
            evidence = 0.35 * evidence + 0.65 * proximity * (1.0 if possession == ARGENTINA else -1.0)
    return {
        "argentina_players": counts[ARGENTINA],
        "switzerland_players": counts[SWITZERLAND],
        "ball_detected": bool(balls),
        "possession_proxy": possession,
        "evidence": float(np.clip(evidence, -1.0, 1.0)),
    }


def live_probabilities(prematch: np.ndarray, smoothed_evidence: float) -> np.ndarray:
    """Convert bounded visual evidence to an explicitly heuristic probability display."""
    logits = np.log(np.clip(prematch, 1e-9, 1.0))
    strength = 0.72
    logits += np.array([
        strength * smoothed_evidence,
        0.16 * (1.0 - abs(smoothed_evidence)),
        -strength * smoothed_evidence,
    ])
    logits -= logits.max()
    values = np.exp(logits)
    return values / values.sum()


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def detect_video(
    source: Path,
    model_path: Path,
    output_jsonl: Path,
    stride: int = 2,
    image_size: int = 960,
    max_frames: int | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(str(model_path))
    device = _device()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sampled = 0
    detected_people = 0
    detected_balls = 0
    with output_jsonl.open("w", encoding="utf-8") as output:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames is not None and frame_index >= max_frames):
                break
            if frame_index % stride:
                frame_index += 1
                continue
            result = model.predict(
                frame,
                classes=[0, 32],
                conf=0.08,
                iou=0.50,
                imgsz=image_size,
                device=device,
                verbose=False,
            )[0]
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confidences = result.boxes.conf.detach().cpu().numpy()
                for box_array, class_id, confidence_value in zip(boxes, classes, confidences, strict=True):
                    box = [round(float(value), 2) for value in box_array]
                    confidence = float(confidence_value)
                    if class_id == 0:
                        if confidence < 0.22 or not is_pitch_person(box, frame.shape):
                            continue
                        team, color_scores = classify_jersey(frame, box)
                        detected_people += 1
                    else:
                        if confidence < 0.08:
                            continue
                        team, color_scores = "BALL", {}
                        detected_balls += 1
                    detections.append({
                        "class_id": int(class_id),
                        "confidence": round(confidence, 4),
                        "box": box,
                        "team": team,
                        "color_scores": color_scores,
                    })
            context = frame_context(detections, frame.shape)
            output.write(json.dumps({
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 4),
                "detections": detections,
                "context": context,
            }) + "\n")
            sampled += 1
            if sampled % 100 == 0:
                elapsed = time.perf_counter() - start
                print(f"detected {sampled} sampled frames ({sampled / elapsed:.2f} infer fps)", flush=True)
            frame_index += 1
    capture.release()
    elapsed = time.perf_counter() - start
    summary = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "device": device,
        "width": width,
        "height": height,
        "fps": fps,
        "source_frames": total_frames,
        "processed_source_frames": min(total_frames, max_frames) if max_frames else total_frames,
        "inference_stride": stride,
        "sampled_frames": sampled,
        "image_size": image_size,
        "person_detections": detected_people,
        "ball_detections": detected_balls,
        "elapsed_seconds": elapsed,
        "sampled_inference_fps": sampled / elapsed if elapsed else 0.0,
    }
    output_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def read_detection_cache(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[int(row["frame_index"])] = row
    if not records:
        raise ValueError(f"no detections in {path}")
    return records


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    context: dict[str, Any],
    prematch: np.ndarray,
    live: np.ndarray,
    timestamp: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = width / 1920.0
    for item in detections:
        x1, y1, x2, y2 = [int(value) for value in item["box"]]
        if item["class_id"] == 32:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = max(8, int(14 * scale))
            cv2.circle(frame, center, radius, COLORS["BALL"], max(2, int(4 * scale)))
            _put_text(frame, f"BALL {item['confidence']:.2f}", (x1, max(30, y1 - 8)), 0.55 * scale, COLORS["BALL"])
        else:
            color = COLORS[item["team"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(3 * scale)))
            label = item["team"] if item["team"] != NEUTRAL else "PERSON"
            _put_text(frame, f"{label} {item['confidence']:.2f}", (x1, max(30, y1 - 8)), 0.48 * scale, color)

    panel_height = int(245 * scale)
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (width, panel_height), (12, 18, 28), -1)
    cv2.addWeighted(panel, 0.83, frame, 0.17, 0, frame)
    _put_text(frame, "SOCCER PROBABILITY ENGINE", (int(35 * scale), int(48 * scale)), 0.88 * scale)
    _put_text(
        frame,
        "CURRENT WIN ODDS - RESEARCH VISUALIZATION, NOT BETTING ADVICE",
        (int(35 * scale), int(82 * scale)),
        0.48 * scale,
        (210, 220, 235),
        max(1, int(1.5 * scale)),
    )
    card_width = int(300 * scale)
    gap = int(24 * scale)
    start_x = int(35 * scale)
    card_y1, card_y2 = int(101 * scale), int(218 * scale)
    card_colors = (COLORS[ARGENTINA], (190, 190, 190), COLORS[SWITZERLAND])
    for index, (name, probability, color) in enumerate(zip(LABELS, live, card_colors, strict=True)):
        x1 = start_x + index * (card_width + gap)
        x2 = x1 + card_width
        cv2.rectangle(frame, (x1, card_y1), (x2, card_y2), (28, 36, 48), -1)
        cv2.rectangle(frame, (x1, card_y1), (x2, card_y2), color, max(2, int(3 * scale)))
        _put_text(frame, name.upper(), (x1 + int(15 * scale), card_y1 + int(34 * scale)), 0.55 * scale, color)
        _put_text(frame, f"{probability * 100:5.1f}%", (x1 + int(15 * scale), card_y1 + int(88 * scale)), 1.25 * scale)
        _put_text(
            frame,
            f"prematch {prematch[index] * 100:.1f}%",
            (x1 + int(15 * scale), card_y1 + int(110 * scale)),
            0.38 * scale,
            (190, 200, 215),
            max(1, int(1.5 * scale)),
        )

    context_x = start_x + 3 * (card_width + gap) + int(8 * scale)
    _put_text(frame, "LIVE DETECTION CONTEXT", (context_x, int(124 * scale)), 0.50 * scale, (210, 220, 235))
    _put_text(
        frame,
        f"ARG players: {context.get('argentina_players', 0)}   SUI players: {context.get('switzerland_players', 0)}",
        (context_x, int(158 * scale)),
        0.46 * scale,
    )
    ball_text = "BALL DETECTED" if context.get("ball_detected") else "ball not detected"
    _put_text(frame, ball_text, (context_x, int(190 * scale)), 0.46 * scale, COLORS["BALL"])
    proxy = context.get("possession_proxy", "unclear")
    _put_text(frame, f"nearest-player proxy: {proxy}", (context_x, int(220 * scale)), 0.42 * scale, (205, 210, 220))

    footer = frame.copy()
    footer_height = int(54 * scale)
    cv2.rectangle(footer, (0, height - footer_height), (width, height), (12, 18, 28), -1)
    cv2.addWeighted(footer, 0.75, frame, 0.25, 0, frame)
    _put_text(
        frame,
        "YOLOv8m COCO: person + sports ball | team colors are heuristic | Source: FIFA / YouTube | classroom presentation",
        (int(28 * scale), height - int(18 * scale)),
        0.42 * scale,
        (220, 225, 235),
        max(1, int(1.5 * scale)),
    )
    _put_text(frame, f"{int(timestamp // 60):02d}:{timestamp % 60:04.1f}", (width - int(135 * scale), height - int(18 * scale)), 0.46 * scale)
    return frame


def render_master(
    source: Path,
    detections_path: Path,
    bundle_path: Path,
    output: Path,
    max_frames: int | None = None,
) -> dict[str, Any]:
    records = read_detection_cache(detections_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    prematch = prematch_probabilities(bundle_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
        "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "fast", "-qp", "0",
        "-pix_fmt", "yuv444p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("failed to open ffmpeg input pipe")
    sorted_indices = sorted(records)
    record_index = 0
    current = records[sorted_indices[0]]
    evidence_window: deque[float] = deque(maxlen=max(1, int(fps * 4)))
    frame_index = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames is not None and frame_index >= max_frames):
                break
            while record_index + 1 < len(sorted_indices) and sorted_indices[record_index + 1] <= frame_index:
                record_index += 1
                current = records[sorted_indices[record_index]]
            evidence_window.append(float(current["context"]["evidence"]))
            smoothed = float(np.mean(evidence_window)) if evidence_window else 0.0
            live = live_probabilities(prematch, smoothed)
            annotated = draw_overlay(
                frame,
                current["detections"],
                current["context"],
                prematch,
                live,
                frame_index / fps,
            )
            process.stdin.write(annotated.tobytes())
            frame_index += 1
            if frame_index % 300 == 0:
                elapsed = time.perf_counter() - start
                print(f"rendered {frame_index}/{total_frames} frames ({frame_index / elapsed:.1f} fps)", flush=True)
    finally:
        capture.release()
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg master render failed with exit code {return_code}")
    summary = {
        "source": str(source),
        "detections": str(detections_path),
        "model_bundle": str(bundle_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "frames": frame_index,
        "fps": fps,
        "resolution": [width, height],
        "prematch_probabilities": {label: float(value) for label, value in zip(LABELS, prematch, strict=True)},
        "elapsed_seconds": time.perf_counter() - start,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,pix_fmt",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def target_video_bitrate(duration: float, target_megabytes: float, audio_kbps: int = 128, safety: float = 0.965) -> int:
    total_kbps = target_megabytes * 1024 * 1024 * 8 * safety / duration / 1000
    return max(250, int(total_kbps - audio_kbps))


def encode_target(source: Path, output: Path, target_megabytes: float) -> dict[str, Any]:
    probe = probe_video(source)
    duration = float(probe["format"]["duration"])
    audio_kbps = 128
    video_kbps = target_video_bitrate(duration, target_megabytes, audio_kbps)
    output.parent.mkdir(parents=True, exist_ok=True)
    passlog = output.parent / f".{output.stem}-2pass"
    common = [
        "-hide_banner", "-loglevel", "warning", "-i", str(source), "-c:v", "libx264", "-preset", "slow",
        "-b:v", f"{video_kbps}k", "-maxrate", f"{video_kbps}k", "-bufsize", f"{video_kbps * 2}k",
        "-passlogfile", str(passlog),
    ]
    subprocess.run(["ffmpeg", "-y", *common, "-pass", "1", "-an", "-f", "null", "/dev/null"], check=True)
    subprocess.run([
        "ffmpeg", "-y", *common, "-pass", "2", "-c:a", "aac", "-b:a", f"{audio_kbps}k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)
    result = {
        "source": str(source),
        "output": str(output),
        "target_megabytes": target_megabytes,
        "actual_bytes": output.stat().st_size,
        "actual_megabytes": output.stat().st_size / (1024 * 1024),
        "video_kbps": video_kbps,
        "audio_kbps": audio_kbps,
        "sha256": sha256_file(output),
        "probe": probe_video(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["actual_megabytes"] > target_megabytes:
        raise RuntimeError(f"{output} exceeds {target_megabytes} MB target")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    detect = subparsers.add_parser("detect")
    detect.add_argument("--source", type=Path, required=True)
    detect.add_argument("--model", type=Path, required=True)
    detect.add_argument("--output", type=Path, required=True)
    detect.add_argument("--stride", type=int, default=2)
    detect.add_argument("--image-size", type=int, default=960)
    detect.add_argument("--max-frames", type=int)
    render = subparsers.add_parser("render")
    render.add_argument("--source", type=Path, required=True)
    render.add_argument("--detections", type=Path, required=True)
    render.add_argument("--bundle", type=Path, default=Path("artifacts/final_run/model.joblib"))
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--max-frames", type=int)
    encode = subparsers.add_parser("encode")
    encode.add_argument("--source", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--target-mb", type=float, required=True)
    args = parser.parse_args()
    if args.command == "detect":
        result = detect_video(args.source, args.model, args.output, args.stride, args.image_size, args.max_frames)
    elif args.command == "render":
        result = render_master(args.source, args.detections, args.bundle, args.output, args.max_frames)
    else:
        result = encode_target(args.source, args.output, args.target_mb)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
