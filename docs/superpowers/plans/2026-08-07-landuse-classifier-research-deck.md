# Landuse classifier research deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Create and verify a concise seven-slide editable research deck about the completed landuse-v1 sentence-classifier study.

**Architecture:** Keep the classifier package unchanged. Author the deck in a temporary JavaScript module using @oai/artifact-tool, export only the final PPTX to slides/, and keep rendered PNGs, montage, layout JSON, source notes, and QA logs temporary. Use the public Hub study files and repository documentation as the evidence source; do not write to Hub, Trackio, Grid'5000, or the external data root.

**Tech Stack:** Node.js ES modules, @oai/artifact-tool, PowerPoint export, the presentation skill's render/overflow helpers, and the local view_image inspection tool.

**Copy gate:** Write every visible sentence for a public research audience in
plain, human language. Remove generic AI filler, inflated claims, internal
planning language, and unnecessary operational jargon. Read every title aloud;
if it sounds like a template or status report, rewrite it before export.

---

### Task 1: Prepare the evidence ledger and deck workspace

**Files:**
- Create: slides/landuse-classifier-research.pptx (final binary deliverable)
- Temporary only: $TMP_DIR/deck.mjs
- Temporary only: $TMP_DIR/source-notes.txt

- [ ] **Step 1: Create an isolated temporary workspace.**

Run:

~~~bash
TMP_DIR="$(mktemp -d "$PWD/.tmp-landuse-classifier-slides.XXXXXX")"
printf '%s\n' "$TMP_DIR"
~~~

All generated intermediate files must stay there and the directory must not be
used as a project deliverable directory.

- [ ] **Step 2: Record the exact evidence values before authoring.**

Write $TMP_DIR/source-notes.txt with these values from the public
studies/landuse-v1/results.json and study.json files:

~~~text
Study: landuse-v1
Dataset: NoeFlandre/osm-polygon-wikidata-sentence-relevance
Dataset revision: 07e421a3020127ced2c19304645a6f63e6735966
Model: jhu-clsp/mmBERT-small
Model revision: abc32620dd4f6ab06f5fbe905dc25f310618e09f
Source commit: 496de7e5fec1b08d92e9bf295b78340224e134e0
Screening seed: 42
Replication seeds: 43, 44
Selection metric: positive-class F1 (eval_f1)
Tie-break: macro-F1 (eval_macro_f1)
No held-out test set: true
a00 screening positive F1: 0.5298
a06 screening positive F1: 0.5640
a06 replication positive F1: 0.6660, 0.6537
a06 mean positive F1: 0.6279
a06 mean balanced accuracy: 0.7796
Baseline mean positive F1: 0.5394
Mean positive-F1 difference: +0.0885
~~~

Record these source URLs in the same file:

~~~text
https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/study.json
https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/results.json
https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/README.md
https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance
https://huggingface.co/jhu-clsp/mmBERT-small
https://github.com/NoeFlandre/slides-colloquium
~~~

- [ ] **Step 3: Initialize the artifact-tool workspace.**

Run:

~~~bash
SKILL_DIR="/Users/noeflandre/.codex/plugins/cache/openai-primary-runtime/presentations/26.805.11740/skills/presentations"
node "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" \
  --workspace "$TMP_DIR"
~~~

Expected result: the temporary workspace has the bundled
@oai/artifact-tool runtime available to an ES module.

### Task 2: Author the seven-slide editable deck

**Files:**
- Create: $TMP_DIR/deck.mjs
- Read: docs/superpowers/specs/2026-08-07-landuse-classifier-research-deck-design.md
- Read: README.md, docs/guides/data-policy.md, docs/architecture/overview.md

- [ ] **Step 1: Define the deck constants and reusable helpers.**

Use Presentation.create with slideSize width 1280 and height 720 and:

~~~js
const COLORS = {
  ink: "#172033", muted: "#5d6b7e", accent: "#2f5d8c",
  accentSoft: "#e8f0f7", teal: "#147d83", coral: "#c85b54",
  paper: "#fbfcfe", white: "#ffffff", line: "#d8e0e8",
};
const SLIDE = { width: 1280, height: 720 };
const FRAME = { left: 72, top: 58, width: 1136, height: 604 };
~~~

Create only small helpers for transparent text boxes, takeaway titles, eyebrow
labels, footer/page markers, rules, and speaker notes. Do not create a general
component system or import a template from the reference repository.

- [ ] **Step 2: Add the title, task, leakage, and cleaning slides.**

Use the reference's title-sidebar silhouette for slide 1 and natural visible
copy:

~~~text
OSM polygon descriptions contain a usable landuse signal
Sentence-level classification with leakage-aware evaluation
Noe Flandre · August 2026
~~~

Slide 2 uses two columns: input is an OSM polygon-description sentence and the
output is landuse-relevant/not landuse-relevant. Explain in plain language why
we need to recognize landuse language before using the polygon description.

Slide 3 uses stacked rows:

~~~text
Pass 1 — discover sentence-content hashes with contradictory labels
Pass 2 — emit one clean representative per remaining usable hash
Split assignment stays polygon-based; the cleaned dataset is never written
~~~

Include a “no polygon leakage” callout. State in plain language that a polygon
can contain sentences with different labels, while the same sentence content
with contradictory labels is excluded. Cite README.md and
docs/guides/data-policy.md in notes.

- [ ] **Step 3: Add the study design slide.**

Use a 50/50 composition. Left-side visible protocol:

~~~text
13 total runs
7 screening runs · seed 42
6 replications · seeds 43 and 44
Selection: positive-class F1
~~~

Right-side visible factors:

~~~text
Context: 128 / 256 / 512 tokens
Learning rate: 1e-4 / 3e-4 / 1e-3
Loss: standard / class-balanced
Training depth: head / last two layers
~~~

Add jhu-clsp/mmBERT-small, frozen-encoder baseline, and binary head. Cite
study.json and the public model card.

- [ ] **Step 4: Add the main result slide with one native chart.**

Create a horizontal bar chart with categories
["baseline · head · 256", "head · 128", "head · 512", "head · lr 1e-4", "head · lr 1e-3", "balanced head", "last two layers · 256"]
and values [0.5298, 0.5062, 0.5233, 0.5287, 0.5144, 0.4919, 0.5640].
Highlight only the final bar with COLORS.teal. Label it
“Positive-class F1 · validation only” and add
“a06-last2-256 · screening F1 0.5640; replicated mean F1 0.6279”.
Cite results.json and the public study README.

- [ ] **Step 5: Add the caveat and closing slides.**

Slide 6 uses two columns:

~~~text
Positive F1 across a06 seeds: 0.5640 → 0.6660
Balanced accuracy across a06 seeds: 0.7681 → 0.7875
Promising validation evidence
No held-out test set yet
Seed and split variation remain visible
Precision/recall trade-offs still need task-level calibration
~~~

Use a restrained coral caution rule; do not claim state-of-the-art or production
readiness.

Slide 7 closes with:

~~~text
1. Freeze the protocol and publish a held-out evaluation split
2. Compare threshold behavior, not only a single operating point
3. Keep model, study registry, and Trackio artifacts linked by identity
~~~

Include the public registry URL and pinned dataset/model/source identity as
supporting text.

- [ ] **Step 6: Add notes and export the final PPTX.**

Every slide must call slide.speakerNotes.textFrame.setText with a short talking
point followed by a literal [Sources] block and full URLs. Export only to:

~~~text
slides/landuse-classifier-research.pptx
~~~

The module must not export intermediate files into slides/.

### Task 3: Render and inspect every slide

**Files:**
- Read: slides/landuse-classifier-research.pptx
- Temporary only: $TMP_DIR/rendered/, $TMP_DIR/montage.webp, $TMP_DIR/qa.txt

- [ ] **Step 1: Render and run overflow checks.**

Run:

~~~bash
SKILL_DIR="/Users/noeflandre/.codex/plugins/cache/openai-primary-runtime/presentations/26.805.11740/skills/presentations"
python "$SKILL_DIR/container_tools/render_slides.py" slides/landuse-classifier-research.pptx
python "$SKILL_DIR/container_tools/create_montage.py" \
  --input_dir slides/landuse-classifier-research \
  --output_file "$TMP_DIR/montage.webp"
python "$SKILL_DIR/container_tools/slides_test.py" slides/landuse-classifier-research.pptx
~~~

Expected result: seven PNGs, a montage, and zero overflow warnings.

- [ ] **Step 2: Inspect the montage and every slide at full size.**

Use view_image on the montage and each slide. Check title wrapping, chart
legibility, margins, footer consistency, unintended overlap/clipping, the
visible distinction between validation and held-out evidence, and whether the
copy sounds like a human researcher rather than a generated template. If any
check fails, revise only $TMP_DIR/deck.mjs and repeat the complete render loop.

- [ ] **Step 3: Inspect deck structure and speaker notes.**

Run a Node inspection using presentation.inspect with kind
slide,textbox,shape,chart,notes. Confirm seven slides, seven visible notes blocks
containing [Sources], a native result chart, and stable names for title/footer
objects.

### Task 4: Final repository QA and delivery

**Files:**
- Read: slides/landuse-classifier-research.pptx
- Read: docs/superpowers/specs/2026-08-07-landuse-classifier-research-deck-design.md

- [ ] **Step 1: Check generated-file scope.**

Run:

~~~bash
git diff --check
git status --short --branch
find slides -maxdepth 2 -type f -print | sort
~~~

Only the intended final deck may be new under slides/; temporary files must not
be tracked.

- [ ] **Step 2: Verify acceptance criteria.**

Confirm seven slides, 16:9 dimensions, exact published metric values,
validation-only wording, cited notes, no copied reference assets, and zero
remote side effects.

- [ ] **Step 3: Commit the final deliverable.**

Run:

~~~bash
git add slides/landuse-classifier-research.pptx
git commit -m "docs: add landuse classifier research deck"
~~~

Do not push, modify the linked reference repository, or touch Grid'5000, HF, or
Trackio state as part of this task.
