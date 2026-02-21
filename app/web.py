"""Flask web interface for face indexing and search."""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.face_index import SUPPORTED_EXTENSIONS, build_index
from app.face_match import find_ranked_matches

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "static" / "uploads"
IMAGES_DIR = UPLOAD_ROOT / "images"
QUERIES_DIR = UPLOAD_ROOT / "queries"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


@app.get("/")
def index_page():
    return render_template("index.html", matches=[], indexed_count=0)


@app.post("/index")
def upload_and_index():
    files = request.files.getlist("images")
    if not files:
        return render_template(
            "index.html",
            error="Не выбраны файлы для индексации.",
            matches=[],
            indexed_count=0,
        )

    shutil.rmtree(IMAGES_DIR, ignore_errors=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for file in files:
        if not file.filename:
            continue
        if not _allowed(file.filename):
            continue

        safe_name = secure_filename(file.filename)
        target = IMAGES_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        file.save(str(target))
        saved += 1

    if saved == 0:
        return render_template(
            "index.html",
            error="Нет валидных изображений (.jpg/.jpeg/.png).",
            matches=[],
            indexed_count=0,
        )

    try:
        index_data = build_index(str(IMAGES_DIR))
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Ошибка индексации: {exc}",
            matches=[],
            indexed_count=0,
        )

    return render_template(
        "index.html",
        success=f"Индексация завершена. Найдено лиц: {len(index_data)}",
        matches=[],
        indexed_count=len(index_data),
    )


@app.post("/search")
def search():
    query = request.files.get("query")
    threshold_raw = request.form.get("threshold", "0.6")

    try:
        threshold = float(threshold_raw)
    except ValueError:
        threshold = 0.6

    if query is None or not query.filename:
        return render_template(
            "index.html",
            error="Не выбрано фото для поиска.",
            matches=[],
            indexed_count=0,
            threshold=threshold,
        )

    if not _allowed(query.filename):
        return render_template(
            "index.html",
            error="Фото запроса должно быть .jpg/.jpeg/.png.",
            matches=[],
            indexed_count=0,
            threshold=threshold,
        )

    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    query_name = secure_filename(query.filename)
    query_path = QUERIES_DIR / f"{uuid.uuid4().hex}_{query_name}"
    query.save(str(query_path))

    if not IMAGES_DIR.exists():
        return render_template(
            "index.html",
            error="Сначала загрузите набор изображений для индексации.",
            matches=[],
            indexed_count=0,
            threshold=threshold,
        )

    try:
        index_data = build_index(str(IMAGES_DIR))
        ranked = find_ranked_matches(str(query_path), index_data, threshold=threshold)
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Ошибка поиска: {exc}",
            matches=[],
            indexed_count=0,
            threshold=threshold,
        )

    matches = [
        {
            "path": item["path"],
            "distance": item["distance"],
            "url": url_for("static", filename="uploads/images/" + Path(item["path"]).name),
        }
        for item in ranked
    ]

    query_url = url_for("static", filename="uploads/queries/" + query_path.name)

    return render_template(
        "index.html",
        matches=matches,
        indexed_count=len(index_data),
        threshold=threshold,
        query_url=query_url,
        success=f"Найдено совпадений: {len(matches)}",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
