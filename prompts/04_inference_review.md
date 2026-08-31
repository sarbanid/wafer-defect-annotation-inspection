# Step 4 — Inference workflow with a human-review gate

Turn the trained model into a repeatable pipeline for new inspection
images, with low-confidence predictions routed back to a human.

---

Build a Roboflow Workflow that runs the trained
wafer-optical-defects model on new images, and route any
prediction with confidence below 90% into a "needs review" output
instead of auto-passing it. Give me the Workflow's inference endpoint
and an example curl request I can call from our inspection pipeline.

---

### Notes
- Route "needs review" images back into the Label Studio project from
  Step 2 to close the loop — those become next month's training data.
- Ask Codex to log false negatives (missed defects) separately from
  false positives; on a manufacturing line those have very different
  costs, and it's worth tracking them as separate metrics.
- If images can't leave company-controlled infrastructure, ask
  specifically whether the Workflow can run via Roboflow's
  self-hosted inference server instead of the hosted endpoint.
