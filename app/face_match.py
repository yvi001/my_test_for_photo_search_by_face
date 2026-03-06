"""Low-level face matching against a list of indexed Face objects."""

from __future__ import annotations

from pathlib import Path

from app.models import Face

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None


def _require_face_recognition() -> None:
    if face_recognition is None:
        raise RuntimeError(
            "The 'face_recognition' package is required. Install from requirements.txt"
        )


def _extract_query_encoding(query_image: str):
    query_path = Path(query_image)
    if not query_path.exists() or not query_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_image}")
    image = face_recognition.load_image_file(str(query_path))
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face found in query image")
    return encodings[0]


def find_ranked_matches(
    query_image: str,
    index: list[Face],
    threshold: float = 0.6,
) -> list[dict]:
    """Return ranked unique photo matches: [{"path", "distance", "face_id"}, ...]"""
    _require_face_recognition()
    query_encoding = _extract_query_encoding(query_image)
    return find_ranked_matches_from_encoding(query_encoding, index, threshold)


def find_ranked_matches_from_encoding(
    query_encoding,
    index: list[Face],
    threshold: float = 0.6,
) -> list[dict]:
    """Same as find_ranked_matches but accepts a pre-computed encoding."""
    _require_face_recognition()

    best_distance: dict[str, float] = {}
    best_face_id: dict[str, str] = {}

    for face in index:
        distance = float(face_recognition.face_distance([face.embedding], query_encoding)[0])
        if distance <= threshold:
            prev = best_distance.get(face.image_path)
            if prev is None or distance < prev:
                best_distance[face.image_path] = distance
                best_face_id[face.image_path] = face.id

    return [
        {"path": path, "distance": dist, "face_id": best_face_id[path]}
        for path, dist in sorted(best_distance.items(), key=lambda kv: kv[1])
    ]


def find_matches(query_image: str, index: list[Face], threshold: float = 0.6) -> list[str]:
    """Convenience wrapper returning just paths (used by CLI)."""
    return [r["path"] for r in find_ranked_matches(query_image, index, threshold)]
