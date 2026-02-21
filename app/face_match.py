"""Face matching against an in-memory face index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import face_recognition
except ImportError:  # pragma: no cover - depends on runtime environment
    face_recognition = None


def _require_face_recognition() -> None:
    if face_recognition is None:
        raise RuntimeError(
            "The 'face_recognition' package is required. Install dependencies from requirements.txt"
        )


def find_matches(query_image: str, index: list[dict[str, Any]], threshold: float = 0.6) -> list[str]:
    """Return image paths whose indexed faces match the query face by threshold."""

    _require_face_recognition()

    query_path = Path(query_image)
    if not query_path.exists() or not query_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_image}")

    image = face_recognition.load_image_file(str(query_path))
    query_encodings = face_recognition.face_encodings(image)

    if not query_encodings:
        raise ValueError("No face found in query image")

    query_encoding = query_encodings[0]

    matches: list[str] = []
    seen: set[str] = set()

    for item in index:
        indexed_encoding = item["encoding"]
        distance = face_recognition.face_distance([indexed_encoding], query_encoding)[0]

        if distance <= threshold:
            path = item["path"]
            if path not in seen:
                seen.add(path)
                matches.append(path)

    return matches
