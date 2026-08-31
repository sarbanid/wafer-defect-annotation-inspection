# Step 2 — Human annotation & QA in Label Studio

Auto-labels from Roboflow are a starting point, not ground truth.
Route them through Label Studio for a human review pass before
training on them.

Start Label Studio first: `label-studio start --port 8080`

---

Create a Label Studio project called "wafer-defect-review" for
image annotation with a bounding-box labeling config matching these
classes: scratch, contamination, chip-crack, solder-void,
misalignment. Import the auto-labeled tasks from
./data/annotated/roboflow_export.json as pre-annotations so reviewers
only need to correct, not draw from scratch. Once I've finished
reviewing, tell me how many tasks are labeled versus still pending in
that project.

---

### Notes
- Export Roboflow's auto-labels to `./data/annotated/` first (ask
  Codex: "export the current annotations from the Roboflow project as
  a Label Studio-compatible JSON").
- For pixel-level defect outlines instead of boxes, ask for a polygon
  or vector labeling config instead of bounding boxes.
- Keep this step entirely on internal infra — do not route sensitive
  component images through it if `label-studio` isn't running on
  company-controlled infrastructure.
