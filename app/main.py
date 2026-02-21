"""CLI for indexing photos and searching by face."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from app.face_index import build_index
from app.face_match import find_matches


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photo search by face")
    parser.add_argument("--images-dir", required=True, help="Directory with images to index")
    parser.add_argument("--query-image", required=True, help="Path to query image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Face match distance threshold (lower is stricter, default: 0.6)",
    )
    parser.add_argument("--output", help="Optional path to save JSON output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        index = build_index(args.images_dir)
        matches = find_matches(args.query_image, index, threshold=args.threshold)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Matches:")
    if matches:
        for path in matches:
            print(f" - {path}")
    else:
        print(" (none)")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"matches": matches}, f, ensure_ascii=False, indent=2)
        print(f"Saved results to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
