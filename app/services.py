"""Service layer: business logic orchestrating storage, jobs, and core algorithms."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.models import Cluster, Face, Job, Person, Photo, Project
from app.storage import Database

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class ProjectService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, name: str, description: str = "") -> Project:
        if not name.strip():
            raise ValueError("Project name cannot be empty.")
        project = Project(name=name.strip(), description=description.strip())
        self.db.create_project(project)
        return project

    def list_all(self) -> list[Project]:
        return self.db.list_projects()

    def get(self, project_id: str) -> Project:
        p = self.db.get_project(project_id)
        if p is None:
            raise KeyError(f"Project not found: {project_id}")
        return p

    def update(self, project_id: str, name: str, description: str, settings: dict | None = None) -> Project:
        p = self.get(project_id)
        p.name = name.strip()
        p.description = description.strip()
        if settings is not None:
            p.settings.update(settings)
        self.db.update_project(p)
        return p

    def delete(self, project_id: str, data_dir: Path) -> None:
        import shutil
        self.db.delete_project(project_id)
        project_dir = data_dir / "projects" / project_id
        shutil.rmtree(project_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Photo
# ---------------------------------------------------------------------------

class PhotoService:
    def __init__(self, db: Database, data_dir: Path) -> None:
        self.db = db
        self.data_dir = data_dir

    def _project_dirs(self, project_id: str) -> tuple[Path, Path, Path]:
        base = self.data_dir / "projects" / project_id
        photos_dir = base / "photos"
        thumbs_dir = base / "thumbs"
        faces_dir = base / "faces"
        for d in (photos_dir, thumbs_dir, faces_dir):
            d.mkdir(parents=True, exist_ok=True)
        return photos_dir, thumbs_dir, faces_dir

    def add_photos(self, project_id: str, files: list) -> tuple[list[Photo], int]:
        """Save uploaded files to disk and create Photo records.

        Returns (saved_photos, skipped_count).
        """
        from PIL import Image
        from werkzeug.utils import secure_filename

        photos_dir, thumbs_dir, _ = self._project_dirs(project_id)
        saved: list[Photo] = []
        skipped = 0

        for file in files:
            if not getattr(file, "filename", None):
                skipped += 1
                continue
            suffix = Path(file.filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue

            photo_id = uuid.uuid4().hex
            safe_name = secure_filename(file.filename)
            stored_path = photos_dir / f"{photo_id}_{safe_name}"
            file.save(str(stored_path))

            try:
                img = Image.open(str(stored_path)).convert("RGB")
                width, height = img.size
                thumb_path = thumbs_dir / f"{photo_id}.jpg"
                img.thumbnail((200, 200))
                img.save(str(thumb_path), "JPEG", quality=85)
                img.close()
                thumb_rel = str(thumb_path.relative_to(self.data_dir))
            except Exception:
                thumb_rel = None
                width = height = 0

            photo = Photo(
                id=photo_id,
                project_id=project_id,
                original_filename=file.filename,
                stored_path=str(stored_path.relative_to(self.data_dir)),
                thumbnail_path=thumb_rel,
                width=width,
                height=height,
                file_size=stored_path.stat().st_size,
                status="pending",
            )
            self.db.add_photo(photo)
            saved.append(photo)

        return saved, skipped

    def delete(self, photo_id: str) -> None:
        photo = self.db.get_photo(photo_id)
        if photo is None:
            raise KeyError(f"Photo not found: {photo_id}")
        self.db.delete_faces_by_photo(photo_id)
        self.db.delete_photo(photo_id)
        # Remove files from disk
        for rel_path in (photo.stored_path, photo.thumbnail_path):
            if rel_path:
                p = self.data_dir / rel_path
                p.unlink(missing_ok=True)

    def get_absolute_path(self, rel_path: str) -> Path:
        return self.data_dir / rel_path


# ---------------------------------------------------------------------------
# Extraction (runs in background thread via JobQueue)
# ---------------------------------------------------------------------------

class ExtractionService:
    """Extracts face embeddings and thumbnails for a list of photos."""

    def __init__(self, db: Database, data_dir: Path, detection_model: str = "cnn") -> None:
        self.db = db
        self.data_dir = data_dir
        self.detection_model = detection_model

    def run(self, job: Job, photo_ids: list[str]) -> None:
        """Work function – called by JobQueue in background thread."""
        job.total = len(photo_ids)
        self.db.update_job(job)

        for i, photo_id in enumerate(photo_ids):
            photo = self.db.get_photo(photo_id)
            if photo is None:
                continue

            photo.status = "processing"
            self.db.update_photo(photo)

            try:
                faces = self._extract(photo)
                self.db.add_faces(faces)
                photo.status = "indexed"
                photo.face_count = len(faces)
                photo.error = None
            except Exception as exc:
                logger.warning("Extraction failed for photo %s: %s", photo_id, exc)
                photo.status = "failed"
                photo.error = str(exc)

            self.db.update_photo(photo)
            job.current_item = i + 1
            job.progress = int((i + 1) / job.total * 100)
            self.db.update_job(job)

    def _extract(self, photo: Photo) -> list[Face]:
        try:
            import face_recognition
        except ImportError as exc:
            raise RuntimeError("face_recognition package not installed") from exc

        from PIL import Image, ImageOps
        import numpy as np

        abs_path = self.data_dir / photo.stored_path
        pil_img = Image.open(str(abs_path))
        pil_img = ImageOps.exif_transpose(pil_img)
        image = np.array(pil_img.convert("RGB"))
        upsample = 1 if self.detection_model == "cnn" else 2
        locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model=self.detection_model)
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)

        faces_dir = self.data_dir / "projects" / photo.project_id / "faces"
        faces_dir.mkdir(parents=True, exist_ok=True)

        faces: list[Face] = []
        pil_image = Image.open(str(abs_path)).convert("RGB")
        img_w, img_h = pil_image.size

        for loc, enc in zip(locations, encodings):
            face_id = uuid.uuid4().hex
            top, right, bottom, left = loc

            # Crop face with padding
            pad = max(10, int((bottom - top) * 0.2))
            crop_box = (
                max(0, left - pad),
                max(0, top - pad),
                min(img_w, right + pad),
                min(img_h, bottom + pad),
            )
            face_img = pil_image.crop(crop_box)
            face_img.thumbnail((150, 150))
            thumb_path = faces_dir / f"{face_id}.jpg"
            face_img.save(str(thumb_path), "JPEG", quality=85)
            face_img.close()

            faces.append(Face(
                id=face_id,
                photo_id=photo.id,
                project_id=photo.project_id,
                image_path=str(abs_path),
                embedding=enc,
                bbox=(top, right, bottom, left),
                thumbnail_path=str(thumb_path.relative_to(self.data_dir)),
            ))

        pil_image.close()
        return faces


def make_extraction_work(photo_ids: list[str], data_dir: Path, detection_model: str = "cnn"):
    """Return a work_fn ready for JobQueue.submit."""
    from functools import partial
    return partial(_extraction_work, photo_ids=photo_ids, data_dir=data_dir, detection_model=detection_model)


def _extraction_work(job: Job, db: Database, *, photo_ids: list[str], data_dir: Path, detection_model: str = "cnn") -> None:
    ExtractionService(db, data_dir, detection_model=detection_model).run(job, photo_ids)


# ---------------------------------------------------------------------------
# Clustering (runs as background job)
# ---------------------------------------------------------------------------

class ClusterService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def run(self, job: Job, project_id: str, eps: float = 0.5, min_samples: int = 2) -> None:
        """Work function – called by JobQueue in background thread."""
        try:
            import numpy as np
            from sklearn.cluster import DBSCAN
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required: pip install scikit-learn") from exc

        faces = self.db.list_faces(project_id)
        if not faces:
            raise ValueError("No faces indexed in this project.")

        job.total = len(faces)
        self.db.update_job(job)

        embeddings = np.array([f.embedding for f in faces])
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(embeddings)

        cluster_map: dict[int, list[Face]] = {}
        for face, label in zip(faces, labels):
            if label == -1:
                continue
            cluster_map.setdefault(label, []).append(face)

        clusters: list[Cluster] = []
        for cluster_faces in cluster_map.values():
            centroid = np.mean([f.embedding for f in cluster_faces], axis=0)
            clusters.append(Cluster(
                project_id=project_id,
                face_ids=[f.id for f in cluster_faces],
                centroid=centroid,
            ))

        self.db.clear_clusters(project_id)
        if clusters:
            self.db.save_clusters(clusters)

        job.result = {
            "cluster_count": len(clusters),
            "noise_count": int(np.sum(labels == -1)),
        }
        job.current_item = len(faces)
        job.progress = 100
        self.db.update_job(job)

        logger.info("Clustering: %d clusters, %d noise faces.", len(clusters), job.result["noise_count"])


def make_cluster_work(project_id: str, eps: float, min_samples: int):
    from functools import partial
    return partial(_cluster_work, project_id=project_id, eps=eps, min_samples=min_samples)


def _cluster_work(job: Job, db: Database, *, project_id: str, eps: float, min_samples: int) -> None:
    ClusterService(db).run(job, project_id, eps=eps, min_samples=min_samples)


# ---------------------------------------------------------------------------
# Person management
# ---------------------------------------------------------------------------

class PersonService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, project_id: str, name: str, **kwargs) -> Person:
        if not name.strip():
            raise ValueError("Person name cannot be empty.")
        person = Person(project_id=project_id, name=name.strip(), **kwargs)
        self.db.create_person(person)
        return person

    def update(self, person_id: str, **kwargs) -> Person:
        person = self.db.get_person(person_id)
        if person is None:
            raise KeyError(f"Person not found: {person_id}")
        for k, v in kwargs.items():
            setattr(person, k, v)
        self.db.update_person(person)
        return person

    def delete(self, person_id: str) -> None:
        self.db.delete_person(person_id)

    def assign_face(self, face_id: str, person_id: Optional[str]) -> None:
        """Assign or unassign a face to a person."""
        face = self.db.get_face(face_id)
        if face is None:
            raise KeyError(f"Face not found: {face_id}")
        face.person_id = person_id
        self.db.update_face(face)

    def assign_cluster(self, cluster_id: str, person_id: str) -> None:
        """Assign all faces in a cluster to a person."""
        cluster = self.db.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        for face_id in cluster.face_ids:
            face = self.db.get_face(face_id)
            if face:
                face.person_id = person_id
                self.db.update_face(face)
        cluster.person_id = person_id
        self.db.update_cluster(cluster)

    def set_avatar(self, person_id: str, face_id: str) -> None:
        person = self.db.get_person(person_id)
        if person is None:
            raise KeyError(f"Person not found: {person_id}")
        person.avatar_face_id = face_id
        self.db.update_person(person)

    def merge(self, target_person_id: str, source_person_id: str) -> None:
        """Move all faces from source person into target person, then delete source."""
        source = self.db.get_person(source_person_id)
        if source is None:
            raise KeyError(f"Source person not found: {source_person_id}")
        for face in self.db.list_faces(source.project_id):
            if face.person_id == source_person_id:
                face.person_id = target_person_id
                self.db.update_face(face)
        self.db.delete_person(source_person_id)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def find_by_image(
        self,
        query_image_path: str,
        project_ids: list[str],
        threshold: float = 0.6,
        face_index: Optional[int] = None,
        detection_model: str = "cnn",
    ) -> dict:
        """Search across one or more projects by a query image file.

        *face_index*: which detected face in the query image to use (0-based).
        Returns {query_faces, matches, project_ids_searched}.
        """
        try:
            import face_recognition
        except ImportError as exc:
            raise RuntimeError("face_recognition package not installed") from exc

        from app.face_match import find_ranked_matches

        from PIL import Image, ImageOps
        import numpy as np

        pil_img = Image.open(query_image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        image = np.array(pil_img.convert("RGB"))
        upsample = 1 if detection_model == "cnn" else 2
        locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model=detection_model)
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)

        if not encodings:
            raise ValueError("No face detected in the query image.")

        # Build a temporary Face list for find_ranked_matches
        query_faces = [
            {"index": i, "bbox": loc, "encoding": enc}
            for i, (loc, enc) in enumerate(zip(locations, encodings))
        ]

        selected_idx = face_index if face_index is not None and 0 <= face_index < len(encodings) else 0
        selected_enc = encodings[selected_idx]

        # Load indexed faces from requested projects
        all_faces: list[Face] = []
        for pid in project_ids:
            all_faces.extend(self.db.list_faces(pid))

        if not all_faces:
            return {"query_faces": query_faces, "matches": [], "project_ids_searched": project_ids}

        matches = find_ranked_matches_from_encoding(selected_enc, all_faces, threshold)
        return {
            "query_faces": query_faces,
            "matches": matches,
            "project_ids_searched": project_ids,
            "selected_face_index": selected_idx,
        }

    def find_by_person(self, person_id: str, project_ids: list[str], threshold: float = 0.6) -> list[dict]:
        """Find all photos (across projects) that contain faces similar to a person's faces."""
        import numpy as np

        person = self.db.get_person(person_id)
        if person is None:
            raise KeyError(f"Person not found: {person_id}")

        person_faces = [f for f in self.db.list_faces(person.project_id) if f.person_id == person_id]
        if not person_faces:
            return []

        # Use the centroid of the person's faces as the query
        centroid = np.mean([f.embedding for f in person_faces], axis=0)

        all_faces: list[Face] = []
        for pid in project_ids:
            all_faces.extend(self.db.list_faces(pid))

        return find_ranked_matches_from_encoding(centroid, all_faces, threshold)


# ---------------------------------------------------------------------------
# Standalone helpers used by SearchService
# ---------------------------------------------------------------------------

def find_ranked_matches_from_encoding(
    query_encoding,
    index: list[Face],
    threshold: float = 0.6,
) -> list[dict]:
    """Core vector search: given a raw encoding, return ranked photo matches."""
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError("face_recognition package not installed") from exc

    best: dict[str, float] = {}
    best_face: dict[str, str] = {}

    for face in index:
        distance = float(face_recognition.face_distance([face.embedding], query_encoding)[0])
        if distance <= threshold:
            prev = best.get(face.image_path)
            if prev is None or distance < prev:
                best[face.image_path] = distance
                best_face[face.image_path] = face.id

    return [
        {"path": path, "distance": dist, "face_id": best_face[path]}
        for path, dist in sorted(best.items(), key=lambda kv: kv[1])
    ]
