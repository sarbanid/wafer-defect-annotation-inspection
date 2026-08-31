"""Rewrite Label Studio task image URLs from file:/// to /data/local-files/."""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
from pathlib import Path

DB = Path(
    os.environ.get(
        "LABEL_STUDIO_DB",
        str(Path.home() / "AppData" / "Local" / "label-studio" / "label-studio" / "label_studio.sqlite3"),
    )
)
DOC_ROOT = Path(
    os.environ.get(
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
        str(Path(__file__).resolve().parents[1] / "data" / "annotated"),
    )
)
IMAGES_DIR = DOC_ROOT / "wafer-optical-defects-coco" / "train"
PROJECT_ID = 2


def file_url_to_local_files(url: str) -> str | None:
    if not url:
        return None
    if url.startswith("/data/local-files/"):
        return url

    path: Path | None = None
    if url.startswith("file:"):
        # file:///C:/... or file:///C%3A/...
        parsed = urllib.parse.urlparse(url)
        raw = urllib.parse.unquote(parsed.path)
        # On Windows urlparse gives /C:/...
        if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
        path = Path(raw)
    else:
        path = Path(urllib.parse.unquote(url))

    name = path.name
    candidate = IMAGES_DIR / name
    if not candidate.exists():
        # try original path if it already includes train/
        if path.exists():
            candidate = path
        else:
            return None

    rel = candidate.resolve().relative_to(DOC_ROOT.resolve()).as_posix()
    return f"/data/local-files/?d={rel}"


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, data FROM task WHERE project_id = ?",
        (PROJECT_ID,),
    ).fetchall()

    updated = 0
    missing = 0
    unchanged = 0
    samples: list[str] = []

    for row in rows:
        data = json.loads(row["data"])
        old = data.get("image", "")
        new = file_url_to_local_files(old)
        if new is None:
            missing += 1
            if len(samples) < 3:
                samples.append(f"MISSING id={row['id']} old={old}")
            continue
        if new == old:
            unchanged += 1
            continue
        data["image"] = new
        cur.execute(
            "UPDATE task SET data = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), row["id"]),
        )
        updated += 1
        if updated <= 2:
            samples.append(f"OK id={row['id']} -> {new}")

    conn.commit()
    conn.close()
    print(f"total={len(rows)} updated={updated} unchanged={unchanged} missing={missing}")
    for s in samples:
        print(s)


if __name__ == "__main__":
    main()
