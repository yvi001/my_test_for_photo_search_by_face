from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import app.web as web
except ModuleNotFoundError:  # pragma: no cover
    web = None


@unittest.skipIf(web is None, "Flask is not installed in the test environment")
class TestWebUI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        upload_root = Path(self.tmpdir.name) / "uploads"

        self.images_dir = upload_root / "images"
        self.queries_dir = upload_root / "queries"

        self.images_patch = patch.object(web, "IMAGES_DIR", self.images_dir)
        self.queries_patch = patch.object(web, "QUERIES_DIR", self.queries_dir)
        self.images_patch.start()
        self.queries_patch.start()
        self.addCleanup(self.images_patch.stop)
        self.addCleanup(self.queries_patch.stop)

        self.client = web.app.test_client()

    def test_index_requires_files(self) -> None:
        response = self.client.post("/index", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Не выбраны файлы".encode("utf-8"), response.data)

    def test_search_returns_ranked_results(self) -> None:
        with patch.object(web, "build_index") as build_index, patch.object(
            web, "find_ranked_matches"
        ) as find_ranked:
            build_index.return_value = [{"path": str(self.images_dir / "a.jpg"), "encoding": [0.1]}]
            find_ranked.return_value = [{"path": str(self.images_dir / "a.jpg"), "distance": 0.2}]

            self.images_dir.mkdir(parents=True, exist_ok=True)
            (self.images_dir / "a.jpg").write_bytes(b"x")

            response = self.client.post(
                "/search",
                data={"threshold": "0.6", "query": (io.BytesIO(b"abc"), "query.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Найдено совпадений: 1".encode("utf-8"), response.data)


if __name__ == "__main__":
    unittest.main()
