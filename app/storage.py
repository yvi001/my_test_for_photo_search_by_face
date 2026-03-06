"""Storage layer: Database protocol and SQLite implementation.

To swap the database backend, implement the Database protocol with a new class
(e.g. PostgreSQLDatabase) and pass it wherever Database is expected.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from app.models import Cluster, Face, Job, Person, Photo, Project

# ---------------------------------------------------------------------------
# Protocol – the only interface consumers depend on
# ---------------------------------------------------------------------------

@runtime_checkable
class Database(Protocol):

    # --- Projects ---
    def create_project(self, project: Project) -> None: ...
    def get_project(self, id: str) -> Optional[Project]: ...
    def list_projects(self) -> list[Project]: ...
    def update_project(self, project: Project) -> None: ...
    def delete_project(self, id: str) -> None: ...

    # --- Photos ---
    def add_photo(self, photo: Photo) -> None: ...
    def get_photo(self, id: str) -> Optional[Photo]: ...
    def list_photos(self, project_id: str, status: Optional[str] = None) -> list[Photo]: ...
    def update_photo(self, photo: Photo) -> None: ...
    def delete_photo(self, id: str) -> None: ...

    # --- Faces ---
    def add_faces(self, faces: list[Face]) -> None: ...
    def get_face(self, id: str) -> Optional[Face]: ...
    def list_faces(self, project_id: str) -> list[Face]: ...
    def list_faces_by_photo(self, photo_id: str) -> list[Face]: ...
    def update_face(self, face: Face) -> None: ...
    def delete_faces_by_photo(self, photo_id: str) -> None: ...

    # --- Persons ---
    def create_person(self, person: Person) -> None: ...
    def get_person(self, id: str) -> Optional[Person]: ...
    def list_persons(self, project_id: Optional[str] = None) -> list[Person]: ...
    def update_person(self, person: Person) -> None: ...
    def delete_person(self, id: str) -> None: ...

    # --- Clusters ---
    def save_clusters(self, clusters: list[Cluster]) -> None: ...
    def list_clusters(self, project_id: str) -> list[Cluster]: ...
    def get_cluster(self, id: str) -> Optional[Cluster]: ...
    def update_cluster(self, cluster: Cluster) -> None: ...
    def clear_clusters(self, project_id: str) -> None: ...

    # --- Jobs ---
    def create_job(self, job: Job) -> None: ...
    def get_job(self, id: str) -> Optional[Job]: ...
    def update_job(self, job: Job) -> None: ...
    def list_jobs(self, project_id: str) -> list[Job]: ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    settings    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS photos (
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path       TEXT NOT NULL,
    thumbnail_path    TEXT,
    uploaded_at       TEXT NOT NULL,
    width             INTEGER NOT NULL DEFAULT 0,
    height            INTEGER NOT NULL DEFAULT 0,
    file_size         INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL DEFAULT 'pending',
    error             TEXT,
    face_count        INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS faces (
    id             TEXT PRIMARY KEY,
    photo_id       TEXT NOT NULL,
    project_id     TEXT NOT NULL,
    image_path     TEXT NOT NULL,
    embedding      BLOB NOT NULL,
    detected_at    TEXT NOT NULL,
    person_id      TEXT,
    bbox_top       INTEGER,
    bbox_right     INTEGER,
    bbox_bottom    INTEGER,
    bbox_left      INTEGER,
    thumbnail_path TEXT,
    FOREIGN KEY (photo_id)    REFERENCES photos(id)   ON DELETE CASCADE,
    FOREIGN KEY (project_id)  REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS persons (
    id             TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    name           TEXT NOT NULL,
    notes          TEXT NOT NULL DEFAULT '',
    date_of_birth  TEXT,
    tags           TEXT NOT NULL DEFAULT '[]',
    avatar_face_id TEXT,
    metadata       TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS clusters (
    id         TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    person_id  TEXT,
    centroid   BLOB NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cluster_faces (
    cluster_id TEXT NOT NULL,
    face_id    TEXT NOT NULL,
    PRIMARY KEY (cluster_id, face_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    type         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',
    progress     INTEGER NOT NULL DEFAULT 0,
    total        INTEGER NOT NULL DEFAULT 0,
    current_item INTEGER NOT NULL DEFAULT 0,
    result       TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
"""


class SQLiteDatabase:
    """SQLite-backed implementation of the Database protocol."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # --- Projects ---

    def create_project(self, p: Project) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, created_at, settings) VALUES (?,?,?,?,?)",
                (p.id, p.name, p.description, p.created_at.isoformat(), json.dumps(p.settings)),
            )

    def get_project(self, id: str) -> Optional[Project]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (id,)).fetchone()
        return _row_to_project(row) if row else None

    def list_projects(self) -> list[Project]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [_row_to_project(r) for r in rows]

    def update_project(self, p: Project) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE projects SET name=?, description=?, settings=? WHERE id=?",
                (p.name, p.description, json.dumps(p.settings), p.id),
            )

    def delete_project(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (id,))

    # --- Photos ---

    def add_photo(self, photo: Photo) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO photos (id,project_id,original_filename,stored_path,thumbnail_path,"
                "uploaded_at,width,height,file_size,status,error,face_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    photo.id, photo.project_id, photo.original_filename, photo.stored_path,
                    photo.thumbnail_path, photo.uploaded_at.isoformat(),
                    photo.width, photo.height, photo.file_size,
                    photo.status, photo.error, photo.face_count,
                ),
            )

    def get_photo(self, id: str) -> Optional[Photo]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM photos WHERE id=?", (id,)).fetchone()
        return _row_to_photo(row) if row else None

    def list_photos(self, project_id: str, status: Optional[str] = None) -> list[Photo]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM photos WHERE project_id=? AND status=? ORDER BY uploaded_at DESC",
                    (project_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM photos WHERE project_id=? ORDER BY uploaded_at DESC",
                    (project_id,),
                ).fetchall()
        return [_row_to_photo(r) for r in rows]

    def update_photo(self, photo: Photo) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE photos SET status=?,error=?,face_count=?,thumbnail_path=? WHERE id=?",
                (photo.status, photo.error, photo.face_count, photo.thumbnail_path, photo.id),
            )

    def delete_photo(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM photos WHERE id=?", (id,))

    # --- Faces ---

    def add_faces(self, faces: list[Face]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO faces (id,photo_id,project_id,image_path,embedding,"
                "detected_at,person_id,bbox_top,bbox_right,bbox_bottom,bbox_left,thumbnail_path) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [_face_to_row(f) for f in faces],
            )

    def get_face(self, id: str) -> Optional[Face]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM faces WHERE id=?", (id,)).fetchone()
        return _row_to_face(row) if row else None

    def list_faces(self, project_id: str) -> list[Face]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM faces WHERE project_id=? ORDER BY detected_at",
                (project_id,),
            ).fetchall()
        return [_row_to_face(r) for r in rows]

    def list_faces_by_photo(self, photo_id: str) -> list[Face]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM faces WHERE photo_id=?", (photo_id,)).fetchall()
        return [_row_to_face(r) for r in rows]

    def update_face(self, face: Face) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE faces SET person_id=?, thumbnail_path=? WHERE id=?",
                (face.person_id, face.thumbnail_path, face.id),
            )

    def delete_faces_by_photo(self, photo_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))

    # --- Persons ---

    def create_person(self, person: Person) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO persons (id,project_id,name,notes,date_of_birth,tags,avatar_face_id,metadata) "
                "VALUES (?,?,?,?,?,?,?,?)",
                _person_to_row(person),
            )

    def get_person(self, id: str) -> Optional[Person]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM persons WHERE id=?", (id,)).fetchone()
        return _row_to_person(row) if row else None

    def list_persons(self, project_id: Optional[str] = None) -> list[Person]:
        with self._connect() as conn:
            if project_id:
                rows = conn.execute(
                    "SELECT * FROM persons WHERE project_id=? ORDER BY name", (project_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM persons ORDER BY name").fetchall()
        return [_row_to_person(r) for r in rows]

    def update_person(self, person: Person) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE persons SET name=?,notes=?,date_of_birth=?,tags=?,avatar_face_id=?,metadata=? WHERE id=?",
                (
                    person.name, person.notes,
                    person.date_of_birth.isoformat() if person.date_of_birth else None,
                    json.dumps(person.tags), person.avatar_face_id,
                    json.dumps(person.metadata), person.id,
                ),
            )

    def delete_person(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE faces SET person_id=NULL WHERE person_id=?", (id,))
            conn.execute("DELETE FROM persons WHERE id=?", (id,))

    # --- Clusters ---

    def save_clusters(self, clusters: list[Cluster]) -> None:
        if not clusters:
            return
        project_id = clusters[0].project_id
        with self._connect() as conn:
            conn.execute("DELETE FROM cluster_faces WHERE cluster_id IN "
                         "(SELECT id FROM clusters WHERE project_id=?)", (project_id,))
            conn.execute("DELETE FROM clusters WHERE project_id=?", (project_id,))
            for c in clusters:
                conn.execute(
                    "INSERT INTO clusters (id,project_id,person_id,centroid) VALUES (?,?,?,?)",
                    (c.id, c.project_id, c.person_id, pickle.dumps(c.centroid)),
                )
                conn.executemany(
                    "INSERT INTO cluster_faces (cluster_id,face_id) VALUES (?,?)",
                    [(c.id, fid) for fid in c.face_ids],
                )

    def list_clusters(self, project_id: str) -> list[Cluster]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM clusters WHERE project_id=?", (project_id,)).fetchall()
            result = []
            for row in rows:
                face_rows = conn.execute(
                    "SELECT face_id FROM cluster_faces WHERE cluster_id=?", (row["id"],)
                ).fetchall()
                result.append(Cluster(
                    id=row["id"], project_id=row["project_id"], person_id=row["person_id"],
                    centroid=pickle.loads(row["centroid"]),
                    face_ids=[fr["face_id"] for fr in face_rows],
                ))
        return result

    def get_cluster(self, id: str) -> Optional[Cluster]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM clusters WHERE id=?", (id,)).fetchone()
            if not row:
                return None
            face_rows = conn.execute(
                "SELECT face_id FROM cluster_faces WHERE cluster_id=?", (id,)
            ).fetchall()
        return Cluster(
            id=row["id"], project_id=row["project_id"], person_id=row["person_id"],
            centroid=pickle.loads(row["centroid"]),
            face_ids=[fr["face_id"] for fr in face_rows],
        )

    def update_cluster(self, cluster: Cluster) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE clusters SET person_id=? WHERE id=?", (cluster.person_id, cluster.id)
            )

    def clear_clusters(self, project_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cluster_faces WHERE cluster_id IN "
                         "(SELECT id FROM clusters WHERE project_id=?)", (project_id,))
            conn.execute("DELETE FROM clusters WHERE project_id=?", (project_id,))

    # --- Jobs ---

    def create_job(self, job: Job) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id,project_id,type,status,progress,total,current_item,"
                "result,error,created_at,finished_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job.id, job.project_id, job.type, job.status, job.progress,
                    job.total, job.current_item,
                    json.dumps(job.result) if job.result else None,
                    job.error, job.created_at.isoformat(),
                    job.finished_at.isoformat() if job.finished_at else None,
                ),
            )

    def get_job(self, id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (id,)).fetchone()
        return _row_to_job(row) if row else None

    def update_job(self, job: Job) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?,progress=?,total=?,current_item=?,result=?,error=?,finished_at=? WHERE id=?",
                (
                    job.status, job.progress, job.total, job.current_item,
                    json.dumps(job.result) if job.result else None,
                    job.error,
                    job.finished_at.isoformat() if job.finished_at else None,
                    job.id,
                ),
            )

    def list_jobs(self, project_id: str) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 20",
                (project_id,),
            ).fetchall()
        return [_row_to_job(r) for r in rows]


# ---------------------------------------------------------------------------
# Row ↔ model helpers
# ---------------------------------------------------------------------------

def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        id=row["id"], name=row["name"], description=row["description"],
        created_at=datetime.fromisoformat(row["created_at"]),
        settings=json.loads(row["settings"]),
    )


def _row_to_photo(row: sqlite3.Row) -> Photo:
    return Photo(
        id=row["id"], project_id=row["project_id"],
        original_filename=row["original_filename"],
        stored_path=row["stored_path"], thumbnail_path=row["thumbnail_path"],
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        width=row["width"], height=row["height"], file_size=row["file_size"],
        status=row["status"], error=row["error"], face_count=row["face_count"],
    )


def _face_to_row(f: Face) -> tuple:
    bbox = f.bbox or (None, None, None, None)
    return (
        f.id, f.photo_id, f.project_id, f.image_path,
        pickle.dumps(f.embedding), f.detected_at.isoformat(),
        f.person_id, bbox[0], bbox[1], bbox[2], bbox[3],
        f.thumbnail_path,
    )


def _row_to_face(row: sqlite3.Row) -> Face:
    bbox = None
    if row["bbox_top"] is not None:
        bbox = (row["bbox_top"], row["bbox_right"], row["bbox_bottom"], row["bbox_left"])
    return Face(
        id=row["id"], photo_id=row["photo_id"], project_id=row["project_id"],
        image_path=row["image_path"],
        embedding=pickle.loads(row["embedding"]),
        detected_at=datetime.fromisoformat(row["detected_at"]),
        person_id=row["person_id"], bbox=bbox,
        thumbnail_path=row["thumbnail_path"],
    )


def _person_to_row(p: Person) -> tuple:
    return (
        p.id, p.project_id, p.name, p.notes,
        p.date_of_birth.isoformat() if p.date_of_birth else None,
        json.dumps(p.tags), p.avatar_face_id, json.dumps(p.metadata),
    )


def _row_to_person(row: sqlite3.Row) -> Person:
    dob = None
    if row["date_of_birth"]:
        dob = date.fromisoformat(row["date_of_birth"])
    return Person(
        id=row["id"], project_id=row["project_id"], name=row["name"],
        notes=row["notes"], date_of_birth=dob,
        tags=json.loads(row["tags"]),
        avatar_face_id=row["avatar_face_id"],
        metadata=json.loads(row["metadata"]),
    )


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"], project_id=row["project_id"], type=row["type"],
        status=row["status"], progress=row["progress"],
        total=row["total"], current_item=row["current_item"],
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )
