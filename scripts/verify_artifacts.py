"""Verify every file recorded in the final-run SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "final_run"
MANIFEST = ARTIFACT_DIR / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> list[str]:
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in recorded.items():
        path = ARTIFACT_DIR / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            errors.append(f"hash mismatch: {relative}")
    unrecorded = sorted(
        str(path.relative_to(ARTIFACT_DIR))
        for path in ARTIFACT_DIR.rglob("*")
        if path.is_file() and path.name != MANIFEST.name and str(path.relative_to(ARTIFACT_DIR)) not in recorded
    )
    errors.extend(f"unrecorded: {relative}" for relative in unrecorded)
    return errors


def main() -> None:
    errors = verify()
    if errors:
        raise SystemExit("artifact verification failed:\n" + "\n".join(f"- {error}" for error in errors))
    print("artifact verification passed")


if __name__ == "__main__":
    main()
