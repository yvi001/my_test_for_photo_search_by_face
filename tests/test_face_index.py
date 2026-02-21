from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import face_index


class TestBuildIndex(unittest.TestCase):
    def test_raises_when_no_valid_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(face_index, "face_recognition") as fr:
                fr.load_image_file.return_value = "img"
                fr.face_encodings.return_value = [[0.1]]
                with self.assertRaises(ValueError):
                    face_index.build_index(tmpdir)

    def test_continues_when_some_images_have_no_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "face.jpg"
            p2 = Path(tmpdir) / "noface.png"
            p1.write_bytes(b"x")
            p2.write_bytes(b"x")

            def fake_load(path: str):
                return path

            def fake_encodings(image):
                if str(p1) in image:
                    return [[0.1, 0.2, 0.3]]
                return []

            with patch.object(face_index, "face_recognition") as fr:
                fr.load_image_file.side_effect = fake_load
                fr.face_encodings.side_effect = fake_encodings

                index = face_index.build_index(tmpdir)

            self.assertEqual(len(index), 1)
            self.assertEqual(index[0]["path"], str(p1))


if __name__ == "__main__":
    unittest.main()
