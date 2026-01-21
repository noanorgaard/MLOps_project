"""Prepare minimal Cloud Build contexts.

This keeps `gcloud builds submit` uploads small (no data/models).

Usage:
    uv run python scripts/prepare_cloudbuild_context.py api
    uv run python scripts/prepare_cloudbuild_context.py drift
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare(kind: str) -> Path:
    """Prepare a minimal build context.

    Args:
        kind: Either "api" or "drift".

    Returns:
        The created context directory.
    """
    if kind not in {"api", "drift"}:
        raise ValueError("kind must be 'api' or 'drift'")

    ctx_dir = ROOT / ".cloudbuild" / kind

    # Keep the Dockerfile/cloudbuild.yaml that already live in ctx_dir, but refresh the copied sources.
    (ctx_dir / "src").mkdir(parents=True, exist_ok=True)

    # Replace src/ completely
    src_dst = ctx_dir / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    src_dst.mkdir(parents=True, exist_ok=True)

    shutil.copytree(ROOT / "src" / "ai_vs_human", src_dst / "ai_vs_human")

    for filename in ["pyproject.toml", "uv.lock", "README.md"]:
        shutil.copy2(ROOT / filename, ctx_dir / filename)

    return ctx_dir


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/prepare_cloudbuild_context.py <api|drift>")

    kind = sys.argv[1]
    ctx_dir = prepare(kind)
    print(ctx_dir)


if __name__ == "__main__":
    main()
