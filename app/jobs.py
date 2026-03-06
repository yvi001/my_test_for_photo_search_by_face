"""Background job queue for heavy operations (face extraction, clustering).

Uses a single-worker ThreadPoolExecutor so that face_recognition/dlib calls
are always serialized — dlib is not safe to call from multiple threads at once.

Usage:
    queue = get_job_queue(db_factory)
    queue.submit(job, work_fn)   # work_fn(job, db) -> None
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Optional

from app.models import Job
from app.storage import Database

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, db_factory: Callable[[], Database]) -> None:
        self._db_factory = db_factory
        # max_workers=1 keeps dlib calls serialized
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-worker")

    def submit(self, job: Job, work: Callable[[Job, Database], None]) -> None:
        """Persist the job record then run *work* in the background thread."""
        db = self._db_factory()
        db.create_job(job)

        def _run() -> None:
            thread_db = self._db_factory()
            j = thread_db.get_job(job.id)
            if j is None:
                return
            j.status = "running"
            thread_db.update_job(j)
            try:
                work(j, thread_db)
                # Re-fetch to get latest progress values written by work()
                j = thread_db.get_job(job.id)
                j.status = "done"
                j.progress = 100
                j.finished_at = datetime.utcnow()
            except Exception as exc:
                logger.exception("Job %s (%s) failed", job.id, job.type)
                j = thread_db.get_job(job.id)
                j.status = "failed"
                j.error = str(exc)
                j.finished_at = datetime.utcnow()
            thread_db.update_job(j)

        self._executor.submit(_run)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_queue: Optional[JobQueue] = None


def init_job_queue(db_factory: Callable[[], Database]) -> JobQueue:
    global _queue
    _queue = JobQueue(db_factory)
    return _queue


def get_job_queue() -> JobQueue:
    if _queue is None:
        raise RuntimeError("Job queue not initialised. Call init_job_queue() first.")
    return _queue
