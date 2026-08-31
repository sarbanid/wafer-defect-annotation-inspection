# Step 3 — Train on the reviewed dataset

Once QA in Label Studio is done, pull the corrected labels back into
Roboflow and train.

---

Import the reviewed annotations from Label Studio project
"wafer-defect-review" back into the Roboflow project
"wafer-optical-defects", generate a new dataset version with an
80/10/10 train/val/test split, and start a training run using
RF-DETR. Report mAP, precision, and recall per class once training
finishes, and call out any class with notably worse recall than the
others.

---

### Notes
- RF-DETR is a solid default for small-object surface defects; ask
  for YOLO instead if you need faster inference on edge/line hardware
  rather than best accuracy.
- If a class has weak recall, the fix is almost always more labeled
  examples for that class, not a different model — check the class
  distribution report before retraining.
