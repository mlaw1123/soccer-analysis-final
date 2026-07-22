"""Verify presentation video hashes, size caps, streams, and resolution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "presentation" / "results" / "run_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height",
        "-of", "json", str(path),
    ], text=True))


def verify() -> list[str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    for expected in manifest["deliverables"]:
        path = ROOT / expected["path"]
        if not path.is_file():
            errors.append(f"missing: {expected['path']}")
            continue
        size = path.stat().st_size
        if size != expected["bytes"]:
            errors.append(f"size mismatch: {expected['path']}")
        if size > expected["target_megabytes"] * 1024 * 1024:
            errors.append(f"over target: {expected['path']}")
        if sha256(path) != expected["sha256"]:
            errors.append(f"hash mismatch: {expected['path']}")
        metadata = probe(path)
        streams = metadata.get("streams", [])
        video = [stream for stream in streams if stream.get("codec_type") == "video"]
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if not video or video[0].get("codec_name") != "h264":
            errors.append(f"missing H.264 video: {expected['path']}")
        elif (video[0].get("width"), video[0].get("height")) != (1920, 1080):
            errors.append(f"resolution mismatch: {expected['path']}")
        if not audio or audio[0].get("codec_name") != "aac":
            errors.append(f"missing AAC audio: {expected['path']}")
    return errors


def main() -> None:
    errors = verify()
    if errors:
        raise SystemExit("presentation verification failed:\n" + "\n".join(f"- {error}" for error in errors))
    print("presentation verification passed")


if __name__ == "__main__":
    main()
