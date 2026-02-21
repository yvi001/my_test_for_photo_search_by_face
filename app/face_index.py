"""Utilities for building a face-embedding index from an image directory."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

try:
    import face_recognition
except ImportError:  # pragma: no cover - depends on runtime environment
    face_recognition = None

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logger = logging.getLogger(__name__)


def _require_face_recognition() -> None:
    if face_recognition is None:
        raise RuntimeError(
            "The 'face_recognition' package is required. Install dependencies from requirements.txt"
        )


def build_index(images_dir: str) -> list[dict[str, Any]]:
    """Build a face index from all images in a directory.

    Returns a list of dictionaries with keys:
      - path: str path to image
      - encoding: face embedding object/array
    """

    _require_face_recognition()

    directory = Path(images_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    image_paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        raise ValueError(f"No valid images found in directory: {images_dir}")

    index: list[dict[str, Any]] = []

    for image_path in image_paths:
        try:
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)
        except Exception as exc:
            logger.warning("Skipping '%s': failed to process image (%s)", image_path, exc)
            continue

        if not encodings:
            logger.warning("No face detected in image: %s", image_path)
            continue

        for encoding in encodings:
            index.append({"path": str(image_path), "encoding": encoding})

    if not index:
        raise ValueError(
            "No faces were indexed. Ensure the directory contains images with visible faces."
        )

    return index
