from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import face_match


class TestFindMatches(unittest.TestCase):
    def test_query_file_not_found(self) -> None:
        with patch.object(face_match, "face_recognition"):
            with self.assertRaises(FileNotFoundError):
                face_match.find_matches("missing.jpg", index=[])

    def test_no_face_in_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            query = Path(tmpdir) / "query.jpg"
            query.write_bytes(b"x")

            with patch.object(face_match, "face_recognition") as fr:
                fr.load_image_file.return_value = "img"
                fr.face_encodings.return_value = []

                with self.assertRaises(ValueError):
                    face_match.find_matches(str(query), index=[])

    def test_ranked_matches_sorted_by_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            query = Path(tmpdir) / "query.jpg"
            query.write_bytes(b"x")
            index = [
                {"path": "img1.jpg", "encoding": [0.1]},
                {"path": "img2.jpg", "encoding": [0.2]},
            ]

            with patch.object(face_match, "face_recognition") as fr:
                fr.load_image_file.return_value = "query"
                fr.face_encodings.return_value = [[0.0]]

                def fake_distance(candidates, _q):
                    return [0.4] if candidates[0] == [0.1] else [0.2]

                fr.face_distance.side_effect = fake_distance

                ranked = face_match.find_ranked_matches(str(query), index, threshold=0.6)

            self.assertEqual([row["path"] for row in ranked], ["img2.jpg", "img1.jpg"])


if __name__ == "__main__":
    unittest.main()
