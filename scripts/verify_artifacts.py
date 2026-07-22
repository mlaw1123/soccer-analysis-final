"""Verify every file recorded in the final-run SHA-256 manifest."""

from __future__ import annotations

from soccer_final.artifacts import verify


def main() -> None:
    errors = verify()
    if errors:
        raise SystemExit("artifact verification failed:\n" + "\n".join(f"- {error}" for error in errors))
    print("artifact verification passed")


if __name__ == "__main__":
    main()
