"""Flask web application – multi-page, HTMX-powered."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from flask import (
    Flask,
    abort,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from app.jobs import init_job_queue, get_job_queue
from app.models import Job
from app.services import (
    ClusterService,
    ExtractionService,
    PersonService,
    PhotoService,
    ProjectService,
    SearchService,
    make_cluster_work,
    make_extraction_work,
)
from app.storage import SQLiteDatabase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", Path(__file__).parent.parent / "data"))
DB_PATH = DATA_DIR / "face_index.db"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB per request
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")


def _db():
    return SQLiteDatabase(DB_PATH)


# Initialise job queue once at import time (one background worker thread)
init_job_queue(_db)


# ---------------------------------------------------------------------------
# File serving (photos and face crops live outside static/)
# ---------------------------------------------------------------------------

@app.get("/files/<path:subpath>")
def serve_file(subpath: str):
    abs_path = DATA_DIR / subpath
    if not abs_path.exists() or not abs_path.is_file():
        abort(404)
    return send_file(str(abs_path))


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

@app.get("/")
def projects_list():
    projects = ProjectService(_db()).list_all()
    return render_template("projects_list.html", projects=projects)


@app.post("/projects")
def project_create():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    try:
        project = ProjectService(_db()).create(name, description)
        return redirect(url_for("project_photos", project_id=project.id))
    except ValueError as exc:
        projects = ProjectService(_db()).list_all()
        return render_template("projects_list.html", projects=projects, error=str(exc))


@app.post("/projects/<project_id>/delete")
def project_delete(project_id: str):
    ProjectService(_db()).delete(project_id, DATA_DIR)
    return redirect(url_for("projects_list"))


@app.get("/projects/<project_id>/settings")
def project_settings(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    return render_template("project_settings.html", project=project)


@app.post("/projects/<project_id>/settings")
def project_settings_save(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    detection_model = request.form.get("face_detection_model", "cnn")
    if detection_model not in ("cnn", "hog"):
        detection_model = "cnn"
    ProjectService(db).update(
        project_id,
        request.form["name"],
        request.form.get("description", ""),
        settings={"face_detection_model": detection_model},
    )
    return redirect(url_for("project_photos", project_id=project_id))


# ---------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------

@app.get("/projects/<project_id>/photos")
def project_photos(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    photos = db.list_photos(project_id)
    active_jobs = [j for j in db.list_jobs(project_id) if j.status in ("queued", "running")]
    return render_template(
        "project_photos.html",
        project=project,
        photos=photos,
        active_jobs=active_jobs,
    )


@app.post("/projects/<project_id>/photos/upload")
def photos_upload(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    files = request.files.getlist("photos")

    svc = PhotoService(db, DATA_DIR)
    try:
        saved, skipped = svc.add_photos(project_id, files)
    except Exception as exc:
        photos = db.list_photos(project_id)
        return render_template(
            "project_photos.html", project=project, photos=photos,
            active_jobs=[], error=str(exc),
        )

    if not saved:
        photos = db.list_photos(project_id)
        return render_template(
            "project_photos.html", project=project, photos=photos,
            active_jobs=[], error="No valid images uploaded (.jpg/.jpeg/.png).",
        )

    # Queue background extraction
    photo_ids = [p.id for p in saved]
    detection_model = project.settings.get("face_detection_model", "cnn")
    job = Job(project_id=project_id, type="extract", total=len(photo_ids))
    get_job_queue().submit(job, make_extraction_work(photo_ids, DATA_DIR, detection_model=detection_model))

    return redirect(url_for("project_photos", project_id=project_id))


@app.post("/projects/<project_id>/photos/<photo_id>/delete")
def photo_delete(project_id: str, photo_id: str):
    PhotoService(_db(), DATA_DIR).delete(photo_id)
    return redirect(url_for("project_photos", project_id=project_id))


@app.get("/projects/<project_id>/photos/<photo_id>")
def photo_detail(project_id: str, photo_id: str):
    db = _db()
    project = _require_project(db, project_id)
    photo = db.get_photo(photo_id)
    if photo is None:
        abort(404)
    faces = db.list_faces_by_photo(photo_id)
    persons = db.list_persons(project_id)
    person_map = {p.id: p for p in persons}
    return render_template(
        "photo_detail.html",
        project=project, photo=photo, faces=faces, person_map=person_map,
    )


# ---------------------------------------------------------------------------
# Faces browser
# ---------------------------------------------------------------------------

@app.get("/projects/<project_id>/faces")
def faces_browser(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    faces = db.list_faces(project_id)
    persons = db.list_persons(project_id)
    person_map = {p.id: p for p in persons}

    # Group by person_id (None = unassigned)
    unassigned = [f for f in faces if not f.person_id]
    assigned: dict[str, list] = {}
    for f in faces:
        if f.person_id:
            assigned.setdefault(f.person_id, []).append(f)

    return render_template(
        "faces_browser.html",
        project=project, unassigned=unassigned,
        assigned=assigned, person_map=person_map, persons=persons,
    )


@app.post("/projects/<project_id>/faces/<face_id>/assign")
def face_assign(project_id: str, face_id: str):
    person_id = request.form.get("person_id") or None
    PersonService(_db()).assign_face(face_id, person_id)
    # HTMX or redirect
    if request.headers.get("HX-Request"):
        return "", 200
    return redirect(url_for("faces_browser", project_id=project_id))


@app.post("/projects/<project_id>/faces/<face_id>/set-avatar")
def face_set_avatar(project_id: str, face_id: str):
    face = _db().get_face(face_id)
    if face and face.person_id:
        PersonService(_db()).set_avatar(face.person_id, face_id)
    return redirect(url_for("person_detail", project_id=project_id, person_id=face.person_id))


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

@app.get("/projects/<project_id>/cluster")
def cluster_view(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    clusters = db.list_clusters(project_id)
    persons = db.list_persons(project_id)
    person_map = {p.id: p for p in persons}
    face_map = {f.id: f for f in db.list_faces(project_id)}
    active_jobs = [j for j in db.list_jobs(project_id) if j.type == "cluster" and j.status in ("queued", "running")]
    return render_template(
        "cluster_review.html",
        project=project, clusters=clusters,
        persons=persons, person_map=person_map,
        face_map=face_map, active_jobs=active_jobs,
    )


@app.post("/projects/<project_id>/cluster/run")
def cluster_run(project_id: str):
    db = _db()
    _require_project(db, project_id)
    try:
        eps = float(request.form.get("eps", "0.5"))
        min_samples = int(request.form.get("min_samples", "2"))
    except ValueError:
        eps, min_samples = 0.5, 2

    job = Job(project_id=project_id, type="cluster")
    get_job_queue().submit(job, make_cluster_work(project_id, eps, min_samples))
    return redirect(url_for("cluster_view", project_id=project_id))


@app.post("/projects/<project_id>/cluster/<cluster_id>/assign")
def cluster_assign(project_id: str, cluster_id: str):
    person_id = request.form.get("person_id")
    if not person_id:
        return redirect(url_for("cluster_view", project_id=project_id))
    PersonService(_db()).assign_cluster(cluster_id, person_id)
    return redirect(url_for("cluster_view", project_id=project_id))


# ---------------------------------------------------------------------------
# Persons
# ---------------------------------------------------------------------------

@app.get("/projects/<project_id>/persons")
def persons_list(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    persons = db.list_persons(project_id)
    face_map = {f.id: f for f in db.list_faces(project_id)}
    return render_template(
        "persons_list.html", project=project, persons=persons, face_map=face_map,
    )


@app.post("/projects/<project_id>/persons")
def person_create(project_id: str):
    db = _db()
    _require_project(db, project_id)
    name = request.form.get("name", "").strip()
    try:
        person = PersonService(db).create(project_id, name)
        return redirect(url_for("person_detail", project_id=project_id, person_id=person.id))
    except ValueError as exc:
        persons = db.list_persons(project_id)
        project = db.get_project(project_id)
        face_map = {f.id: f for f in db.list_faces(project_id)}
        return render_template(
            "persons_list.html", project=project, persons=persons,
            face_map=face_map, error=str(exc),
        )


@app.get("/projects/<project_id>/persons/<person_id>")
def person_detail(project_id: str, person_id: str):
    db = _db()
    project = _require_project(db, project_id)
    person = db.get_person(person_id)
    if person is None:
        abort(404)
    faces = [f for f in db.list_faces(project_id) if f.person_id == person_id]
    # Collect unique photos containing this person
    photo_ids = list(dict.fromkeys(f.photo_id for f in faces))
    photos = [p for p in (db.get_photo(pid) for pid in photo_ids) if p]
    return render_template(
        "person_detail.html",
        project=project, person=person, faces=faces, photos=photos,
    )


@app.post("/projects/<project_id>/persons/<person_id>/edit")
def person_edit(project_id: str, person_id: str):
    svc = PersonService(_db())
    tags_raw = request.form.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    dob_raw = request.form.get("date_of_birth", "").strip()
    dob = date.fromisoformat(dob_raw) if dob_raw else None
    svc.update(
        person_id,
        name=request.form.get("name", "").strip(),
        notes=request.form.get("notes", ""),
        date_of_birth=dob,
        tags=tags,
    )
    return redirect(url_for("person_detail", project_id=project_id, person_id=person_id))


@app.post("/projects/<project_id>/persons/<person_id>/delete")
def person_delete(project_id: str, person_id: str):
    PersonService(_db()).delete(person_id)
    return redirect(url_for("persons_list", project_id=project_id))


@app.post("/projects/<project_id>/persons/<target_id>/merge")
def person_merge(project_id: str, target_id: str):
    source_id = request.form.get("source_id")
    if source_id:
        PersonService(_db()).merge(target_id, source_id)
    return redirect(url_for("person_detail", project_id=project_id, person_id=target_id))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/search")
def search_global():
    db = _db()
    projects = db.list_projects()
    return render_template("search.html", projects=projects, results=None)


@app.get("/projects/<project_id>/search")
def search_in_project(project_id: str):
    db = _db()
    project = _require_project(db, project_id)
    projects = db.list_projects()
    persons = db.list_persons(project_id)
    return render_template(
        "search.html",
        projects=projects, current_project=project,
        persons=persons, results=None,
    )


@app.post("/search/by-image")
def search_by_image():
    import tempfile, os
    db = _db()
    projects = db.list_projects()

    query_file = request.files.get("query")
    project_ids = request.form.getlist("project_ids") or [p.id for p in projects]
    threshold = _parse_float(request.form.get("threshold"), 0.6)
    face_index = _parse_int(request.form.get("face_index"), None)

    if not query_file or not query_file.filename:
        return render_template("search.html", projects=projects, results=None,
                               error="Please upload a query image.")

    # Save to temp file for processing
    suffix = Path(query_file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        query_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        detection_model = "cnn"
        if len(project_ids) == 1:
            proj = db.get_project(project_ids[0])
            if proj:
                detection_model = proj.settings.get("face_detection_model", "cnn")
        result = SearchService(db).find_by_image(
            tmp_path, project_ids, threshold=threshold, face_index=face_index,
            detection_model=detection_model,
        )
        # Enrich matches with photo objects
        enriched = _enrich_matches(db, result["matches"])
        return render_template(
            "search.html",
            projects=projects, results=enriched,
            query_face_index=result.get("selected_face_index", 0),
            query_faces=result["query_faces"],
            threshold=threshold,
        )
    except ValueError as exc:
        return render_template("search.html", projects=projects, results=None, error=str(exc))
    finally:
        os.unlink(tmp_path)


@app.post("/search/by-person")
def search_by_person():
    db = _db()
    projects = db.list_projects()
    person_id = request.form.get("person_id")
    project_ids = request.form.getlist("project_ids") or [p.id for p in projects]
    threshold = _parse_float(request.form.get("threshold"), 0.6)

    if not person_id:
        return render_template("search.html", projects=projects, results=None,
                               error="Select a person to search by.")
    try:
        matches = SearchService(db).find_by_person(person_id, project_ids, threshold)
        enriched = _enrich_matches(db, matches)
        person = db.get_person(person_id)
        return render_template(
            "search.html",
            projects=projects, results=enriched,
            searched_person=person, threshold=threshold,
        )
    except (KeyError, ValueError) as exc:
        return render_template("search.html", projects=projects, results=None, error=str(exc))


# ---------------------------------------------------------------------------
# Jobs (HTMX polling partials)
# ---------------------------------------------------------------------------

@app.get("/jobs/<job_id>/status")
def job_status(job_id: str):
    job = _db().get_job(job_id)
    if job is None:
        abort(404)
    resp = make_response(render_template("_job_status.html", job=job))
    if job.status == "done" and job.type == "cluster":
        resp.headers["HX-Refresh"] = "true"
    return resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_project(db, project_id: str):
    p = db.get_project(project_id)
    if p is None:
        abort(404)
    return p


def _parse_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _enrich_matches(db, matches: list[dict]) -> list[dict]:
    """Add photo and person objects to match dicts."""
    enriched = []
    for m in matches:
        face = db.get_face(m.get("face_id", ""))
        photo = db.get_photo(face.photo_id) if face else None
        person = db.get_person(face.person_id) if face and face.person_id else None
        enriched.append({**m, "face": face, "photo": photo, "person": person})
    return enriched


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
