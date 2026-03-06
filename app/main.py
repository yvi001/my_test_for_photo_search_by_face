"""CLI for the face search system.

Usage examples:
    python -m app.main project create --name "Holiday 2024"
    python -m app.main project list
    python -m app.main photos add --project <id> --dir ./photos
    python -m app.main index --project <id>
    python -m app.main cluster --project <id> [--eps 0.5] [--min-samples 2]
    python -m app.main search --project <id> --query-image ./query.jpg
    python -m app.main person list --project <id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

APP_DATA_DIR = Path(os.environ.get("APP_DATA_DIR", Path(__file__).parent.parent / "data"))
DB_PATH = APP_DATA_DIR / "face_index.db"


def _get_db():
    from app.storage import SQLiteDatabase
    return SQLiteDatabase(DB_PATH)


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

def cmd_project_create(args) -> int:
    from app.services import ProjectService
    p = ProjectService(_get_db()).create(args.name, args.description or "")
    print(f"Created project: {p.name}  (id: {p.id})")
    return 0


def cmd_project_list(args) -> int:
    projects = _get_db().list_projects()
    if not projects:
        print("No projects found.")
        return 0
    for p in projects:
        print(f"  {p.id}  {p.name}  ({p.created_at:%Y-%m-%d})")
    return 0


def cmd_project_delete(args) -> int:
    from app.services import ProjectService
    ProjectService(_get_db()).delete(args.project, APP_DATA_DIR)
    print(f"Deleted project {args.project}")
    return 0


# ---------------------------------------------------------------------------
# photos
# ---------------------------------------------------------------------------

def cmd_photos_add(args) -> int:
    """Save photos from a directory into a project and queue extraction."""
    from app.jobs import JobQueue
    from app.services import ExtractionService, PhotoService, make_extraction_work

    db = _get_db()
    project = db.get_project(args.project)
    if project is None:
        print(f"Error: project '{args.project}' not found.", file=sys.stderr)
        return 1

    src_dir = Path(args.dir)
    if not src_dir.is_dir():
        print(f"Error: '{args.dir}' is not a directory.", file=sys.stderr)
        return 1

    from app.services import SUPPORTED_EXTENSIONS

    # Wrap files in a simple FileStorage-like adapter
    class _LocalFile:
        def __init__(self, path: Path):
            self._path = path
            self.filename = path.name

        def save(self, dest: str):
            import shutil
            shutil.copy2(str(self._path), dest)

    files = [
        _LocalFile(p)
        for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"No supported images found in '{args.dir}'.", file=sys.stderr)
        return 1

    svc = PhotoService(db, APP_DATA_DIR)
    saved, skipped = svc.add_photos(args.project, files)
    print(f"Saved {len(saved)} photo(s), skipped {skipped}.")

    if not saved:
        return 1

    print("Running face extraction (this may take a while)…")
    photo_ids = [p.id for p in saved]

    from app.models import Job
    job = Job(project_id=args.project, type="extract")

    if args.no_wait:
        # Just queue without a real job queue (CLI runs inline)
        print("Use the web app to monitor progress (--no-wait not yet supported without server).")
    else:
        # Run inline in the CLI process
        from app.services import ExtractionService
        svc2 = ExtractionService(db, APP_DATA_DIR)
        job.status = "running"
        db.create_job(job)
        svc2.run(job, photo_ids)
        job.status = "done"
        job.progress = 100
        db.update_job(job)
        print(f"Extraction complete. Processed {len(saved)} photo(s).")

    return 0


def cmd_photos_list(args) -> int:
    db = _get_db()
    photos = db.list_photos(args.project)
    if not photos:
        print("No photos in this project.")
        return 0
    for p in photos:
        print(f"  [{p.status:10s}] {p.original_filename}  ({p.face_count} faces)  id={p.id}")
    return 0


# ---------------------------------------------------------------------------
# index (re-extract all photos in a project)
# ---------------------------------------------------------------------------

def cmd_index(args) -> int:
    from app.models import Job
    from app.services import ExtractionService

    db = _get_db()
    photos = db.list_photos(args.project)
    if not photos:
        print("No photos found in this project.")
        return 1

    photo_ids = [p.id for p in photos]
    print(f"Re-indexing {len(photo_ids)} photos…")

    job = Job(project_id=args.project, type="extract")
    db.create_job(job)
    ExtractionService(db, APP_DATA_DIR).run(job, photo_ids)
    job.status = "done"
    db.update_job(job)
    print("Done.")
    return 0


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

def cmd_cluster(args) -> int:
    from app.models import Job
    from app.services import ClusterService

    db = _get_db()
    job = Job(project_id=args.project, type="cluster")
    db.create_job(job)
    print(f"Clustering (eps={args.eps}, min_samples={args.min_samples})…")
    ClusterService(db).run(job, args.project, eps=args.eps, min_samples=args.min_samples)
    result = job.result or {}
    print(f"Clusters: {result.get('cluster_count', '?')}  "
          f"Noise (unassigned) faces: {result.get('noise_count', '?')}")
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(args) -> int:
    from app.services import SearchService

    db = _get_db()
    project_ids = [args.project] if args.project else [p.id for p in db.list_projects()]

    try:
        result = SearchService(db).find_by_image(
            args.query_image,
            project_ids,
            threshold=args.threshold,
            face_index=args.face_index,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    matches = result["matches"]
    print(f"Matches ({len(matches)}):")
    for m in matches:
        print(f"  {m['path']}  distance={m['distance']:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {"matches": [{"path": m["path"], "distance": m["distance"]} for m in matches]},
                f, ensure_ascii=False, indent=2,
            )
        print(f"Results saved to: {args.output}")

    return 0


# ---------------------------------------------------------------------------
# person
# ---------------------------------------------------------------------------

def cmd_person_list(args) -> int:
    persons = _get_db().list_persons(args.project if args.project else None)
    if not persons:
        print("No persons found.")
        return 0
    for p in persons:
        print(f"  {p.id}  {p.name}  tags={p.tags}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.main", description="Face search CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # project
    p_proj = sub.add_parser("project", help="Project management")
    p_proj_sub = p_proj.add_subparsers(dest="subcommand", required=True)

    p_create = p_proj_sub.add_parser("create", help="Create a new project")
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--description", default="")

    p_proj_sub.add_parser("list", help="List all projects")

    p_del = p_proj_sub.add_parser("delete", help="Delete a project")
    p_del.add_argument("--project", required=True, metavar="PROJECT_ID")

    # photos
    p_photos = sub.add_parser("photos", help="Photo management")
    p_photos_sub = p_photos.add_subparsers(dest="subcommand", required=True)

    p_add = p_photos_sub.add_parser("add", help="Add photos from a directory")
    p_add.add_argument("--project", required=True, metavar="PROJECT_ID")
    p_add.add_argument("--dir", required=True, metavar="IMAGES_DIR")
    p_add.add_argument("--no-wait", action="store_true", help="Don't wait for extraction")

    p_plist = p_photos_sub.add_parser("list", help="List photos in a project")
    p_plist.add_argument("--project", required=True, metavar="PROJECT_ID")

    # index
    p_idx = sub.add_parser("index", help="Re-extract faces for all photos in a project")
    p_idx.add_argument("--project", required=True, metavar="PROJECT_ID")

    # cluster
    p_cl = sub.add_parser("cluster", help="Cluster faces in a project")
    p_cl.add_argument("--project", required=True, metavar="PROJECT_ID")
    p_cl.add_argument("--eps", type=float, default=0.5)
    p_cl.add_argument("--min-samples", type=int, default=2, dest="min_samples")

    # search
    p_search = sub.add_parser("search", help="Search for similar faces")
    p_search.add_argument("--project", metavar="PROJECT_ID",
                          help="Limit search to this project (default: all)")
    p_search.add_argument("--query-image", required=True, metavar="PATH")
    p_search.add_argument("--threshold", type=float, default=0.6)
    p_search.add_argument("--face-index", type=int, default=0, dest="face_index",
                          help="Which face in the query image to use (0-based)")
    p_search.add_argument("--output", metavar="PATH", help="Save JSON results to file")

    # person
    p_person = sub.add_parser("person", help="Person management")
    p_person_sub = p_person.add_subparsers(dest="subcommand", required=True)
    p_plist2 = p_person_sub.add_parser("list", help="List persons")
    p_plist2.add_argument("--project", metavar="PROJECT_ID",
                          help="Filter by project (default: all)")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        ("project", "create"): cmd_project_create,
        ("project", "list"):   cmd_project_list,
        ("project", "delete"): cmd_project_delete,
        ("photos",  "add"):    cmd_photos_add,
        ("photos",  "list"):   cmd_photos_list,
        ("index",   None):     cmd_index,
        ("cluster", None):     cmd_cluster,
        ("search",  None):     cmd_search,
        ("person",  "list"):   cmd_person_list,
    }

    key = (args.command, getattr(args, "subcommand", None))
    fn = dispatch.get(key)
    if fn is None:
        parser.print_help()
        return 1

    try:
        return fn(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
