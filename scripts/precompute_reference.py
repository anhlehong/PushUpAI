from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.reference import build_expert_reference_cache


DEFAULT_TEMPLATE = ROOT_DIR / "push_up_template.mp4"
DEFAULT_CACHE_DIR = ROOT_DIR / "cache"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute cached MediaPipe reference for the fixed push-up template video."
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_TEMPLATE),
        help="Template video path. Defaults to push_up_template.mp4 in repo root.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory for expert reference artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding cached expert reference even when cache matches.",
    )
    args = parser.parse_args()

    reference = build_expert_reference_cache(
        args.video,
        args.cache_dir,
        force_rebuild=args.force,
    )

    summary = {
        "video_name": reference["video_name"],
        "cache_status": reference.get("cache_status"),
        "template_count": reference.get("template_count", 0),
        "rep_count": len(reference.get("reps", [])),
        "preview_video_path": reference.get("preview_video_path"),
        "quality_passed": reference.get("quality", {}).get("passed"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
