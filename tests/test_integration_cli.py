from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.main import main


class TestCliIntegration(unittest.TestCase):
    def test_end_to_end_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()
            image1 = images_dir / "a.jpg"
            image2 = images_dir / "b.png"
            image1.write_bytes(b"x")
            image2.write_bytes(b"x")

            query = Path(tmpdir) / "query.jpg"
            query.write_bytes(b"x")
            output = Path(tmpdir) / "result.json"

            def fake_load(path: str):
                return path

            def fake_encodings(image):
                image_str = str(image)
                if image_str.endswith("query.jpg"):
                    return [[0.0, 0.0]]
                if image_str.endswith("a.jpg"):
                    return [[0.1, 0.1]]
                if image_str.endswith("b.png"):
                    return [[0.9, 0.9]]
                return []

            def fake_face_distance(candidates, query_encoding):
                candidate = candidates[0]
                if candidate == [0.1, 0.1]:
                    return [0.2]
                return [0.9]

            with patch("app.face_index.face_recognition") as fr_index, patch(
                "app.face_match.face_recognition"
            ) as fr_match:
                fr_index.load_image_file.side_effect = fake_load
                fr_index.face_encodings.side_effect = fake_encodings

                fr_match.load_image_file.side_effect = fake_load
                fr_match.face_encodings.side_effect = fake_encodings
                fr_match.face_distance.side_effect = fake_face_distance

                code = main(
                    [
                        "--images-dir",
                        str(images_dir),
                        "--query-image",
                        str(query),
                        "--threshold",
                        "0.6",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["matches"], [str(image1)])


if __name__ == "__main__":
    unittest.main()
