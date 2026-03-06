"""Low-level face extraction from an image directory.

Used by ExtractionService (per-photo, with bbox and thumbnail generation)
and by the CLI for quick one-shot indexing without a project.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.models import Face

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
logger = logging.getLogger(__name__)


def _require_face_recognition() -> None:
    if face_recognition is None:
        raise RuntimeError(
            "The 'face_recognition' package is required. Install from requirements.txt"
        )


def build_index(images_dir: str, project_id: str = "", photo_id: str = "") -> list[Face]:
    """Extract faces from all images in *images_dir*.

    Returns a list of :class:`~app.models.Face` objects (one per detected face).
    Images where no face is detected are skipped with a warning.
    """
    _require_face_recognition()

    directory = Path(images_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    image_paths = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"No valid images found in: {images_dir}")

    index: list[Face] = []

    for image_path in image_paths:
        try:
            from PIL import Image, ImageOps
            import numpy as np
            pil_img = Image.open(str(image_path))
            pil_img = ImageOps.exif_transpose(pil_img)
            image = np.array(pil_img.convert("RGB"))
            locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)
            encodings = face_recognition.face_encodings(image, known_face_locations=locations)
        except Exception as exc:
            logger.warning("Skipping '%s': %s", image_path, exc)
            continue

        if not encodings:
            logger.warning("No face detected: %s", image_path)
            continue

        for loc, enc in zip(locations, encodings):
            index.append(Face(
                id=uuid.uuid4().hex,
                photo_id=photo_id,
                project_id=project_id,
                image_path=str(image_path),
                embedding=enc,
                bbox=loc,
            ))

    if not index:
        raise ValueError(
            "No faces were indexed. Ensure images contain visible faces."
        )

    return index
