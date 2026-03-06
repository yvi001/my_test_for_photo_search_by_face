"""Domain models for the face search system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import numpy as np


@dataclass
class Project:
    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    # Extensible per-project settings: threshold, cluster_eps, cluster_min_samples, etc.
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class Photo:
    project_id: str
    original_filename: str
    stored_path: str          # path relative to DATA_DIR
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    thumbnail_path: Optional[str] = None   # relative to DATA_DIR
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    width: int = 0
    height: int = 0
    file_size: int = 0
    status: str = "pending"   # pending | processing | indexed | failed
    error: Optional[str] = None
    face_count: int = 0


@dataclass
class Face:
    photo_id: str
    project_id: str
    image_path: str           # absolute path to original photo (for matching)
    embedding: np.ndarray
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    person_id: Optional[str] = None
    bbox: Optional[tuple[int, int, int, int]] = None   # (top, right, bottom, left)
    thumbnail_path: Optional[str] = None  # relative to DATA_DIR


@dataclass
class Person:
    project_id: str
    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    notes: str = ""
    date_of_birth: Optional[date] = None
    tags: list[str] = field(default_factory=list)
    avatar_face_id: Optional[str] = None
    # Arbitrary extra fields for future extension (stored as JSON)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    project_id: str
    face_ids: list[str]
    centroid: np.ndarray
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    person_id: Optional[str] = None


@dataclass
class Job:
    project_id: str
    type: str                 # extract | cluster
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"   # queued | running | done | failed
    progress: int = 0         # 0–100
    total: int = 0
    current_item: int = 0
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
