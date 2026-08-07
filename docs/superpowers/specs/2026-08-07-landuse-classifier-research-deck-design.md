# Landuse classifier research deck

## Communication job

By the end of five minutes, a research audience should understand why the
landuse sentence-classification pipeline is designed around leakage control,
what was varied in the first controlled study, what the validation evidence
shows, and why a held-out evaluation is still required.

## Audience and scope

- Audience: research/colloquium audience.
- Duration: approximately five minutes.
- Format: seven-slide editable PowerPoint deck.
- Scope: the completed `landuse-v1` study and the reproducibility safeguards
  already documented in this repository and its public model registry.
- Exclusions: live pipeline status, unpublished data, remote mutations, and
  claims about held-out test performance.

## Narrative and slide jobs

1. **OSM polygon descriptions contain a usable landuse signal.** Introduce the
   task: classify whether a sentence is relevant to landuse.
2. **The task requires sentence-level labels without polygon leakage.** Explain
   why polygon-based splitting and duplicate handling matter for credible
   evaluation.
3. **Cleaning removes contradictory and cross-split duplicate sentence hashes.**
   Show the two-pass clean-input boundary and its fail-closed label policy.
4. **The study varies one training choice at a time.** Summarize the fixed
   protocol, `jhu-clsp/mmBERT-small`, frozen-head baseline, seven screening
   variants, and two replications of the selected finalists.
5. **Unfreezing the last two encoder layers gives the strongest validation F1.**
   Show one uncluttered comparison of positive-class F1 and balanced accuracy,
   highlighting `a06-last2-256` and its three seeds.
6. **The result is promising but is not yet held-out test evidence.** Make the
   validation-only caveat explicit, including the seed-dependent support and
   the precision/recall trade-off.
7. **The pipeline is reproducible; the next experiment is a held-out
   evaluation.** Close with the pinned dataset/model/source identity, public
   registry, and the next research gate.

The slide titles are written as takeaway statements so the title-only sequence
communicates the argument without speaker notes.

## Copy standard

Visible copy must be written for a public research audience in natural,
human-authored language. Prefer short concrete sentences and familiar words
such as “sentence,” “label,” “split,” and “held-out test.” Avoid generic AI
phrases, inflated claims, empty transitions, dashboard jargon, and internal
production vocabulary that does not help the audience understand the result.
Every slide must sound like something the researcher could say aloud without
editing.

## Visual direction

Use the linked `NoeFlandre/slides-colloquium` repository as a read-only
reference for its neutral colloquium language:

- title-sidebar opening;
- one primary idea per slide;
- restrained navy accent with a soft blue support color;
- Rubik-like heading weight and a calm sans-serif body treatment;
- occasional 50/50 columns and stacked rows rather than dashboard cards;
- one section reset only when it improves pacing;
- persistent, quiet footer with project and slide number;
- citations close to claims, with full source URLs in speaker notes.

The deck will use an editable PPTX implementation with the local presentation
artifact tooling. It will not copy code, assets, or content from the reference
repository.

## Evidence and provenance

All numerical claims come from the public files for
`NoeFlandre/osm-polygon-sentence-classifier`:

- `studies/landuse-v1/study.json` for protocol and pinned revisions;
- `studies/landuse-v1/results.json` for scalar metrics;
- `studies/landuse-v1/README.md` for the human-readable registry and caveats;
- this repository's README and architecture/data-policy documentation for the
  clean-input and operational contracts.

The deck will cite the public dataset, model, study registry, and the linked
slide-template repository in speaker notes. It will state explicitly that the
study has no held-out test set and that all reported values are validation
results.

## Acceptance criteria

- The final deck contains seven slides and is readable at 16:9 presentation
  size.
- Every slide has one clear narrative job and a takeaway title.
- Visible copy is concise, natural, public-facing, and free of generic AI
  filler or internal planning language.
- No title wraps unexpectedly; body text remains at or above the presentation
  skill's minimum sizes.
- The ablation comparison uses the published values exactly, labels validation
  metrics, and does not imply test performance.
- Speaker notes include `[Sources]` blocks for every external claim or asset.
- The final PPTX renders successfully; every slide is inspected individually
  and in a montage; no unintended overflow, clipping, overlap, or unresolved
  placeholder remains.
- Only the final deck is placed in the project deliverable directory;
  intermediate build files remain temporary.
- No remote repository, HF artifact, Trackio run, Grid'5000 job, or external
  data root is modified.
