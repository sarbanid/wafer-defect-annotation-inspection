"""Import Label Studio reviewed annotations into Roboflow, then prepare for training."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path

import roboflow

PROJECT = "wafer-optical-defects"
WORKSPACE = "sarbani-datta"
LS_DB = Path(
    os.environ.get(
        "LABEL_STUDIO_DB",
        str(Path.home() / "AppData" / "Local" / "label-studio" / "label-studio" / "label_studio.sqlite3"),
    )
)
ENV = Path(".env")
MAP_CACHE = Path("data/annotated/roboflow_image_id_map.json")


def load_api_key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("ROBOFLOW_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ROBOFLOW_API_KEY missing in .env")


def rf_filename_from_ls_path(image_url: str) -> str:
    name = Path(image_url.split("?d=")[-1] if "?d=" in image_url else image_url).name
    m = re.match(
        r"^(?P<stem>.+)_(?P<ext>jpe?g|png|bmp|webp)\.rf\.[^.]+?\.(?P=ext)$",
        name,
        re.I,
    )
    if m:
        return f"{m.group('stem')}.{m.group('ext')}"
    return name


def percent_to_voc_box(value: dict, ow: int, oh: int) -> tuple[int, int, int, int]:
    x = float(value["x"]) / 100.0 * ow
    y = float(value["y"]) / 100.0 * oh
    w = float(value["width"]) / 100.0 * ow
    h = float(value["height"]) / 100.0 * oh
    xmin = max(0, int(round(x)))
    ymin = max(0, int(round(y)))
    xmax = min(ow, int(round(x + w)))
    ymax = min(oh, int(round(y + h)))
    if xmax <= xmin:
        xmax = min(ow, xmin + 1)
    if ymax <= ymin:
        ymax = min(oh, ymin + 1)
    return xmin, ymin, xmax, ymax


def to_voc_xml(filename: str, ow: int, oh: int, objects: list) -> str:
    parts = [
        "<annotation>",
        f"  <filename>{filename}</filename>",
        "  <size>",
        f"    <width>{ow}</width>",
        f"    <height>{oh}</height>",
        "    <depth>3</depth>",
        "  </size>",
        "  <segmented>0</segmented>",
    ]
    for name, (xmin, ymin, xmax, ymax) in objects:
        parts.extend(
            [
                "  <object>",
                f"    <name>{name}</name>",
                "    <pose>Unspecified</pose>",
                "    <truncated>0</truncated>",
                "    <difficult>0</difficult>",
                "    <bndbox>",
                f"      <xmin>{xmin}</xmin>",
                f"      <ymin>{ymin}</ymin>",
                f"      <xmax>{xmax}</xmax>",
                f"      <ymax>{ymax}</ymax>",
                "    </bndbox>",
                "  </object>",
            ]
        )
    parts.append("</annotation>")
    return "\n".join(parts)


def export_ls_annotations() -> list[dict]:
    conn = sqlite3.connect(LS_DB)
    rows = conn.execute(
        """
        SELECT t.id, t.data, tc.result
        FROM task t
        JOIN task_completion tc ON tc.task_id = t.id
        WHERE t.project_id = 2 AND tc.was_cancelled = 0
        """
    ).fetchall()
    conn.close()
    out = []
    for task_id, data_raw, result_raw in rows:
        data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
        result = json.loads(result_raw) if isinstance(result_raw, str) else (result_raw or [])
        rf_name = rf_filename_from_ls_path(data.get("image", ""))
        objects = []
        ow = oh = None
        for item in result:
            if item.get("type") != "rectanglelabels":
                continue
            ow = int(item.get("original_width") or ow or 0)
            oh = int(item.get("original_height") or oh or 0)
            labels = (item.get("value") or {}).get("rectanglelabels") or []
            if not labels or not ow or not oh:
                continue
            label = labels[0]
            if label == "solder-void":
                label = "solder void"
            objects.append((label, percent_to_voc_box(item["value"], ow, oh)))
        if ow and oh:
            out.append(
                {
                    "task_id": task_id,
                    "rf_name": rf_name,
                    "objects": objects,
                    "xml": to_voc_xml(rf_name, ow, oh, objects),
                }
            )
    return out


def build_name_map(project) -> dict[str, str]:
    if MAP_CACHE.exists():
        data = json.loads(MAP_CACHE.read_text(encoding="utf-8"))
        if len(data) >= 4000:
            print(f"loaded cached map: {len(data)}", flush=True)
            return data

    mapping: dict[str, str] = {}
    offset = 0
    limit = 100
    while True:
        page = project.search(prompt="*", fields=["id", "name", "filename"], offset=offset, limit=limit)
        images = page if isinstance(page, list) else []
        if not images:
            break
        for img in images:
            name = img.get("name") or img.get("filename") or ""
            if not name or "id" not in img:
                continue
            mapping[name] = img["id"]
            mapping[Path(name).name] = img["id"]
        offset += len(images)
        print(f"indexed {offset}...", flush=True)
        if len(images) < limit:
            break
        time.sleep(0.05)
    MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.write_text(json.dumps(mapping), encoding="utf-8")
    print(f"indexed total unique names={len(mapping)}", flush=True)
    return mapping


def main() -> None:
    key = load_api_key()
    rf = roboflow.Roboflow(api_key=key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    print("Exporting Label Studio annotations...")
    anns = export_ls_annotations()
    print(f"LS annotated tasks: {len(anns)}")
    class_counts = Counter(name for a in anns for name, _ in a["objects"])
    print("class counts", dict(class_counts))

    print("Building Roboflow filename->id map...")
    name_map = build_name_map(project)

    uploaded = 0
    missing = 0
    errors = 0
    missing_samples = []
    image_ids = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for a in anns:
            image_id = name_map.get(a["rf_name"])
            if not image_id:
                missing += 1
                if len(missing_samples) < 8:
                    missing_samples.append(a["rf_name"])
                continue
            xml_path = tmp / f"{Path(a['rf_name']).stem}.xml"
            xml_path.write_text(a["xml"], encoding="utf-8")
            try:
                project.save_annotation(
                    annotation_path=str(xml_path),
                    image_id=image_id,
                    annotation_overwrite=True,
                )
                uploaded += 1
                image_ids.append(image_id)
                if uploaded % 50 == 0:
                    print(f"uploaded {uploaded}...")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"error {a['rf_name']}: {e}")
                time.sleep(0.2)

    print(
        f"done uploaded={uploaded} missing={missing} errors={errors} "
        f"missing_samples={missing_samples}"
    )
    Path("data/annotated/roboflow_imported_image_ids.json").write_text(
        json.dumps(image_ids), encoding="utf-8"
    )
    print(f"saved {len(image_ids)} image ids for dataset accept")


if __name__ == "__main__":
    main()
