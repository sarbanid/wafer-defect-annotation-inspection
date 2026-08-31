"""Convert non-empty Label Studio predictions into editable annotations."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
import urllib.error
import urllib.request
from pathlib import Path

DB = Path(
    os.environ.get(
        "LABEL_STUDIO_DB",
        str(Path.home() / "AppData" / "Local" / "label-studio" / "label-studio" / "label_studio.sqlite3"),
    )
)
BASE = "http://127.0.0.1:8080/api"
PROJECT_ID = 2


def api(key: str, method: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def ensure_region_ids(result: list) -> list:
    out = []
    for item in result:
        region = dict(item)
        if not region.get("id"):
            region["id"] = uuid.uuid4().hex[:10]
        out.append(region)
    return out


def main() -> None:
    key = sqlite3.connect(DB).execute("SELECT key FROM authtoken_token").fetchone()[0]

    # Ensure predictions are shown / used for pre-labeling
    api(
        key,
        "PATCH",
        f"/projects/{PROJECT_ID}",
        {
            "show_collab_predictions": True,
            "reveal_preannotations_interactively": False,
            "model_version": "roboflow-sam3-auto-label",
        },
    )

    conn = sqlite3.connect(DB)
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    ann_table = "annotation" if "annotation" in tables else "task_completion"
    print(f"using annotation table: {ann_table}")

    rows = conn.execute(
        """
        SELECT p.task_id, p.result
        FROM prediction p
        JOIN task t ON t.id = p.task_id
        WHERE t.project_id = ?
        ORDER BY p.task_id
        """,
        (PROJECT_ID,),
    ).fetchall()

    existing_tasks = {
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT task_id FROM {ann_table}"
        ).fetchall()
    }

    created = 0
    skipped_empty = 0
    skipped_existing = 0
    errors = 0

    for task_id, result_raw in rows:
        result = json.loads(result_raw) if isinstance(result_raw, str) else (result_raw or [])
        if not result:
            skipped_empty += 1
            continue

        if task_id in existing_tasks:
            skipped_existing += 1
            continue

        payload = {
            "result": ensure_region_ids(result),
            "was_cancelled": False,
            "ground_truth": False,
        }
        try:
            api(key, "POST", f"/tasks/{task_id}/annotations/", payload)
            created += 1
            existing_tasks.add(task_id)
            if created % 100 == 0:
                print(f"created {created}...")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            errors += 1
            if errors <= 5:
                print(f"task {task_id} error {e.code}: {body[:300]}")

    conn.close()
    print(
        f"done created={created} skipped_empty={skipped_empty} "
        f"skipped_existing={skipped_existing} errors={errors}"
    )


if __name__ == "__main__":
    main()
