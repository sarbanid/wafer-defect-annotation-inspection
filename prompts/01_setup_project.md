# Step 1 — Create the Roboflow project and upload images

Drop raw inspection images into `data/raw_images/` first, then run
`codex` from the project root and paste this prompt (edit the bracketed
parts for your actual defect classes):

---

Create a Roboflow object detection project called
"wafer-optical-defects" for classifying manufacturing defects on
optical components. Defect classes: [scratch, contamination,
chip-crack, solder-void, misalignment]. Zip and upload the images in
./data/raw_images. Run auto-labeling with a foundation model on the
upload, then report how many images got labeled per class and flag
any classes with fewer than 20 examples.

---

### Notes
- Pick **object detection** if you need bounding boxes on the defect
  location; pick **classification** instead if you only need a
  pass/fail or defect-type label per image, no localization.
- If your dataset is small, follow up with:
  > Search Roboflow Universe for a public dataset of similar surface
  > defects and check whether it's worth forking into this project to
  > bootstrap training.
- Confirm class names now — renaming later means re-touching every
  annotation.
