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


if __name__ == "__main__":
    unittest.main()
