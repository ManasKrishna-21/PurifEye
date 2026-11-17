# PURIFEYE — MASTER BUILD PROMPT v2 (ANTIGRAVITY FREE PLAN · CPU-ONLY · ONE-GO)

Open the workspace that already contains the dataset, switch the agent to **Planning mode**, and paste everything below **once** as the first message. The work runs in three parts. Part 2 is run by me in my own terminal, so it costs no agent credits.

**Unattended compute budget (edit this number before pasting): 10 hours.**

---

## 0. ROLE AND MISSION

You are a senior computer-vision engineer, MLOps lead and UI/UX designer building **Purifeye**: an end-to-end platform that detects floating plastic bottles in rivers and waterways from still images, video files and live webcam streams, with models that run on a normal CPU and on edge hardware.

**Mission:** beat the current results below, honestly and reproducibly, then deliver a fast real-time pipeline, edge-ready exports, a polished dashboard, and a learning document.

| Model | Input | Precision | Recall | mAP@50 | Latency (CPU) |
|---|---|---|---|---|---|
| YOLOv8 Nano (custom) | 512×512 | 46.8% | 15.2% | 14.6% | ~59 ms (~17 FPS) |
| AutoML TFLite (baseline) | 512×512 | 55.2% | 48.7% | 38.0% | ~1,770 ms (0.5 FPS) |
| AutoML 3×3 tiled | 3×3 tiles | 56.5% | 59.1% | 54.0% | ~15,000 ms (0.06 FPS) |

**Facts about my setup:** I have **no GPU** (CPU only). The dataset (5,592 Plastic Bottles in Rivers, Pascal VOC, rivers in Indonesia, India, Nepal, Pakistan, Bangladesh and Bosnia) is **already in this workspace**: locate it yourself, and ask me only if you truly cannot find it. I am on Antigravity's **free plan with limited credits**, so efficiency matters as much as quality. I am an intermediate Python developer and I am not comfortable with async/await.

---

## 1. RULES (apply throughout)

1. **uv only.** Use uv project commands (uv init, uv add, uv sync, uv lock, uv run). Never pip, never "uv pip", never requirements-style installs. Dependencies live in pyproject.toml with a lockfile.
2. **Credit discipline (critical).**
   - Produce **one** short Implementation Plan (one page) at the start. No per-stage plans, no per-stage walkthroughs; one final Walkthrough only.
   - Do not narrate. Between steps, say nothing beyond one line. Never paste file contents or long logs into chat; summarise in five lines or fewer.
   - Write each file completely in one write. Do not re-read files you just wrote, do not re-run installs, do not make exploratory web searches; consult documentation only when an error demands it.
   - Debug on **tiny subsets** (for example 50 images, 1 epoch) so failures cost seconds, not hours.
   - **Never poll or wait on long jobs.** Anything expected to run longer than about 10 minutes is launched by me in my own terminal (Part 2), not by you.
   - Browser or screenshot verification happens **once**, in Part 3, with at most one fix iteration.
3. **Resumability.** At the start, write a short **AGENTS.md** (these rules, condensed) and a living **PROGRESS.md** (done, next, decisions made, exact commands, known issues). Update PROGRESS.md after each stage. If a session ends early because credits ran out, a new session must be able to continue by reading only AGENTS.md and PROGRESS.md.
4. **No stopping between stages.** Every "Gate" below is a **self-check**: verify the evidence exists, record it in PROGRESS.md, and continue. Stop only if (a) a blocker survives two fix attempts, (b) a destructive action would be needed, or (c) you reach the end of Part 1 or Part 3. This includes Stage 2: proceed on your own recommendation and flag it in the final summary.
5. **Safety.** Never delete, overwrite or move my dataset, the existing model files, or anything under runs/. Never touch the contents of Plastic-Bottle-Detection-Models/. Do not download anything large without stating size and source first.
6. **Honesty.** Never report a number you did not measure. State split, confidence and IoU thresholds, hardware and thread count beside every metric or latency. If a target is missed, say so and diagnose it. Results from smoke tests must be labelled "SMOKE TEST — not real results" everywhere they appear, including in the dashboard.
7. **Reproducibility.** Fixed seeds, configs in configs/, every run logged with config and metrics. Initialise git with a proper .gitignore (data, weights, run outputs, environment). Commit and tag at the end of Part 1 and Part 3.
8. **Keep my layout** and add to it. The four existing scripts remain the public entry points.
9. **Teaching.** Docstrings and comments on non-obvious code. All teaching content goes into one file, **LearnProject.md** (specified in Stage 12), not into per-stage notes. For concurrency use plain threads and queues, never async/await.

---

## 2. EXECUTION MODEL

**Part 1 — Build (you, one session):** Stages 1–11. Build everything, smoke-test the whole chain on a tiny subset, then stop with a handoff message containing: the single command to start the unattended run, an estimated duration, and what to do if it is interrupted.

**Part 2 — Unattended run (me, my terminal, zero credits):** I run the one-command pipeline (Stage 9). It is resumable: running the same command again continues from the last finished step. I will keep the laptop plugged in and awake.

**Part 3 — Finalise (you, short session):** I send "pipeline finished". Read PROGRESS.md and reports/ only, then do Stage 12: ingest real results, check them against the targets, one browser verification pass of the dashboard, run the tests, finalise README, model card and **LearnProject.md** with real numbers, commit and tag, and give a short final summary.

---

## STAGE 1 — SCAFFOLD AND ENVIRONMENT (CPU)

- Report hardware briefly (OS, CPU model and cores, RAM, free disk, webcam present?). Use it to set thread counts, data-loader workers, batch size and whether images can be cached in RAM.
- Create the uv project and layout (new items marked): detect.py, detect_yolo.py, train_yolo.py, prepare_yolo_dataset.py, Plastic-Bottle-Detection-Models/ (read-only), yolo_dataset/, runs/; **NEW** purifeye/ shared package (config, tiling, merging, metrics, tracking, logging, plot theme), evaluate.py, export_models.py, realtime.py, the one-command pipeline, app/, configs/, tests/, docs/, reports/, AGENTS.md, PROGRESS.md, LearnProject.md.
- Dependencies via uv: Ultralytics (CPU PyTorch build), a TFLite runtime that installs cleanly on my OS, OpenCV, NumPy, pandas, plotting, a tracker, FastAPI with a server, pytest, a linter.
- Link or copy the existing dataset into a git-ignored data folder; never modify the original.
- Add uv-run shortcuts (prepare, train, evaluate, export, pipeline, app).

**Gate:** both model families load in a smoke test.

---

## STAGE 2 — DATA AUDIT (AUTOMATED, SHORT)

One script, one short report in reports/ with contact-sheet images:
- Image count, resolutions, aspect ratios, corrupt files, missing or empty annotations, near-duplicates (perceptual hashing).
- Boxes per image, box size as a fraction of image area, **share of small objects** (drives the tiling and resolution choice).
- Class names, "difficult" and "truncated" flags, odd labels (zero-area, out-of-bounds, swapped coordinates), country breakdown where inferable.
- A clear, evidence-based recommendation on image size, tiling and augmentation. Adopt it without waiting for me.

---

## STAGE 3 — CONVERSION, SPLITS, TILED SUBSET

- **VOC to YOLO (prepare_yolo_dataset.py):** normalise by each image's true size (read from the image), clamp boxes, drop invalid ones with a logged reason, treat empty annotations as background, map classes explicitly. **Prove it:** draw converted labels onto 30 random images and add unit tests. A silent label bug is the first suspect for the very low YOLO recall (15.2%).
- **Leakage-safe splits:** train, validation and a **frozen test split** never used for tuning. Prevent near-duplicates and same-scene images from crossing splits. Save the manifest so every model is scored on identical images. Report counts per split and per country.
- **Tiled training subset (CPU-aware):** overlapped tiles (about 640 px, roughly 20% overlap) plus a downscaled full-image copy; keep a box on a tile only if about 40–50% is visible. Because CPU training is slow, **cap the tiled set** to a sample (for example tiles containing boxes plus a small share of background tiles) so the total training images stay near 1.5× the original count. Validation and test remain full images.
- **Label-quality pass:** after the final model exists, review high-confidence "false positives" for missing annotations, correct a reviewed sample, and report results on both original and cleaned labels.

---

## STAGE 4 — UNIFIED EVALUATION HARNESS AND BASELINES

evaluate.py scores every model identically (AutoML TFLite, AutoML 3×3, every YOLO variant, every export):
- Precision, recall, F1, mAP@50, mAP@50-95, per-size-bucket recall and AP (small, medium, large), PR curve, confusion matrix, and the confidence threshold that maximises F1 (chosen on validation, applied to test).
- Latency protocol: warm-up, median and 95th percentile, fixed thread count, batch size 1, hardware named, model load excluded; preprocessing, inference and post-processing reported separately.
- **Reproduce the baseline rows** on the frozen test split. The AutoML 3×3 row is very slow, so first investigate why the 2.76 MB model takes ~1,770 ms (likely thread or delegate settings) and re-measure with proper threading and batched tiles. If it is still too slow, evaluate the tiled AutoML row on a **fixed 150-image subset of the test split** and report **every** model on that same subset in an extra column so comparisons stay fair. If the old numbers do not reproduce, say why and re-baseline.
- Output a table in the same format as the current "Model Evaluation" table.

---

## STAGE 5 — TRAINING STRATEGY (CPU, TIME-BUDGETED)

Treat YOLOv8n at 46.8 / 15.2 / 14.6 as under-performance to diagnose, not a ceiling. Likely causes, in order: label conversion bugs, too few epochs, 512 px too small for distant bottles, training from scratch instead of COCO-pretrained weights, threshold mismatches.

**Budget first, then plan.** Measure real CPU throughput (about 20 training iterations and per-image inference time), then derive epoch counts from my unattended compute budget. Suggested split of the budget: about 20% ablations, 60–65% final training, 15–20% evaluation, tiling comparison and exports. Print the estimate; adjust if it does not fit.

**CPU-friendly settings:** YOLOv8n as the primary model, COCO-pretrained weights, backbone frozen for the first few epochs, image cache in RAM if memory allows, data-loader workers set from the core count, batch size tuned to RAM, cosine schedule, early-stopping patience, mosaic switched off for the last epochs, last and best weights saved every epoch.

**Short ablation on a fixed 15% subset (6–8 epochs each), one change at a time, judged on validation:**
1. 512 versus 640 px full-image.
2. Full-image versus capped tiled training.
3. Augmentation preset for small, floating, reflective objects (scale and translate, colour shifts for water tone and glare, motion blur, brightness variation, copy-paste for sparse bottles, mild rotation; nothing that breaks water-scene physics).
4. YOLOv8s versus YOLOv8n, only if measured throughput makes it affordable.

Pick the winner automatically by validation mAP@50 and recall, then run **one final long training** with that configuration. Pause and resume must survive closing the IDE, a crash or a reboot. Save curves, confusion matrix and a run registry (CSV) with config, metrics and notes.

**Targets (to be validated, not promised):** beat the best baseline row on the frozen test split — precision above 56.5%, recall above 59.1%, mAP@50 above 54.0% — using the accurate (tiled) operating point at far lower latency than 15 s. Stretch: mAP@50 of 65% with recall above 65%. For the live operating point, reach real-time as far as this CPU allows with mAP@50 clearly above 14.6%, and report the true FPS even if 30 is not reached.

---

## STAGE 6 — SMARTER INFERENCE

- **Overlapped tiled inference** (about 20% overlap, configurable grid), tiles run **as one batch**, merged in full-image coordinates with class-aware NMS and fragment merging.
- **Adaptive tiling:** full image first; tile only when the image is high-resolution or full-image confidence suggests small distant bottles.
- Two named operating points: **Accurate** (tiled or adaptive) and **Live** (single pass at a smaller inference size, fastest CPU backend). Report accuracy and latency for each in one table.
- Tune confidence and NMS on validation; optional test-time augmentation; everything in config.
- **Video logic:** an object tracker gives each bottle a persistent ID, smooths flicker, and counts **unique** bottles; optional counting line for a river section.

---

## STAGE 7 — REAL-TIME PIPELINE (realtime.py)

- Three threads: **capture** (always newest frame), **inference** (latest frame), **render/output**. Single-slot queues that **drop stale frames** so lag never builds. Threads and queues only.
- Cap inference resolution, use the Live operating point and the fastest CPU backend from Stage 8, tiling off by default.
- Telemetry: FPS, capture-to-display latency, inference ms, dropped frames, detections, unique-bottle count. Draw boxes, label pills and telemetry with the same blue design tokens as the dashboard (Sky Blue boxes with a dark halo).
- Sources: webcam, video file, stream URL; clean shutdown, reconnect, graceful handling of a missing camera (fall back to a video file and say so).
- Optional annotated-clip recording and a timestamped detection log (CSV or JSON).
- Sustained-FPS benchmark over several minutes plus 95th-percentile latency, with hardware stated.

---

## STAGE 8 — EDGE EXPORTS (export_models.py)

- Export the best model to **ONNX, OpenVINO (CPU), TFLite (float16 and int8, calibrated on representative training images) and NCNN**. Keep the AutoML MobileNet-SSD .tflite as the ultra-light fallback.
- Run the Stage 4 harness on every export: accuracy after conversion (flag a drop above 2 points of mAP@50), size in MB, CPU latency, memory.
- Recommendation matrix: best for laptop CPU real-time, best for Raspberry Pi (about 320 px, int8), best for maximum accuracy. Minimal "edge runner" that loads an exported model without the training stack. Model card with limitations (geographic bias, turbid water, glare, night scenes, partly submerged bottles) and dataset licence notes.

---

## STAGE 9 — ONE-COMMAND UNATTENDED PIPELINE

Build a single entry point that runs, in order: data preparation (skip if done) → ablations → final training → evaluation of all models on the frozen test split → tiled-inference comparison → exports and benchmarks → themed plots and tables → results files in reports/ that the dashboard reads.
- **Idempotent and resumable:** each step writes a done-marker; rerunning continues where it stopped; training resumes from the last checkpoint.
- **Failure-isolated:** if one step fails, record the error, continue with later steps that do not depend on it, and summarise failures at the end.
- Logs to a file, a compact progress line per step, an estimated time remaining, and a reminder that the machine must stay awake and plugged in.
- A **smoke mode** that runs the whole chain on about 50 images for a couple of minutes. Use it to debug everything in Part 1.

---

## STAGE 10 — DASHBOARD (app/)

Local, one uv command. A **FastAPI backend with ordinary synchronous endpoints** serving a **hand-built HTML, CSS and JavaScript frontend** so the design system can be followed exactly. Live video as an MJPEG stream, telemetry by polling or server-sent events. The dashboard **reads results from reports/ files**, so it works with smoke-test results now and real results later; it shows a visible "SMOKE TEST" banner whenever the data is a smoke run.

Pages: **Live Detect**; **Image Analyzer** (full / tiled / adaptive, confidence and overlap controls); **Video Batch** (annotated output and log download); **Model Lab** (two models side by side, accuracy and latency); **Evaluation** (table, PR curve, confusion matrix, ablation history); **Dataset Explorer** (distributions, samples with ground-truth overlays); **Edge Benchmarks**; **About / Model Card**.

### Design system (follow exactly)

**Role:** act as a senior UI/UX visual designer. Aesthetic: authoritative, eco-industrial AI precision — high-contrast matte dark surfaces, sharp architectural geometry, hardware-style telemetry; never playful or generic.

**Colour architecture (blue palette only; no green or olive anywhere)**

Define the palette once as design tokens (CSS variables for the app and one shared theme object for every saved plot); never hard-code a colour elsewhere:
`#242649` Charcoal Navy · `#222B68` Deep Midnight Indigo · `#23337C` Dark Royal Blue · `#263B84` Deep Sapphire · `#364989` Muted Denim · `#386CAF` Steel Blue · `#65A7D3` Sky Blue · `#B2D3EB` Powder Blue · `#CDDFEF` Ice Blue · `#FFFFFF` Pure White.

Roles:
- Base canvas `#242649`. Cards and panels `#222B68`; raised, hovered or selected surfaces `#23337C`.
- Recessed inputs, slider tracks, telemetry wells: `#242649` inside `#222B68` cards, each with a 1px `#364989` border.
- Primary action `#386CAF` (buttons, active toggles, selected states, focal data blocks) with white text; pressed `#263B84`.
- Reticle and detection accent `#65A7D3` (boxes, crosshair brackets, key data points). Hover illumination, active 1px borders and trendlines `#65A7D3`; focus rings `#B2D3EB`.
- Subtle tints: `#386CAF` at 16–28% opacity for badge fills, active matrix cells and ambient glows.
- Text: primary `#FFFFFF` (headlines, key metrics); secondary `#B2D3EB` (descriptions, axis markers, metadata); emphasised secondary `#CDDFEF` (table headers, hovered labels). Hairlines and grid lines `#364989`.
- Contrast: `#65A7D3` is the lowest-contrast colour allowed for small text; `#386CAF` is never used for small text on dark surfaces. No other hues; express errors and warnings with icon, text label and weight, unless I approve a colour later.

**Typography:** headings Space Grotesk 600/700, letter-spacing −0.02em to −0.03em, fluid clamp() sizing; body and controls Plus Jakarta Sans 400/500/600, line-height 1.6; telemetry, metrics and code JetBrains Mono 500 with tabular numerals on every confidence score, FPS, latency (ms) and axis tick.

**Geometry and motifs:** 6–8px radii on cards and buttons, sharp 2px on bounding boxes and telemetry tags; background coordinate grid (1px `#65A7D3` lines at 4–5% opacity); L-shaped `#65A7D3` corner crosshair brackets around viewports and key metric cards; detection overlays with 2px solid `#65A7D3` border, `rgba(101,167,211,0.14)` fill and a solid `#65A7D3` monospace label pill with dark `#242649` text (class and confidence). Give every box a thin dark `#242649` outer halo so it stays visible on blue-grey water, and offer a high-contrast toggle (Ice Blue or White boxes). Icons: monoline SVG, 1.5px stroke `#65A7D3`, sharp joins, inside containers tinted `#386CAF` at 20%.

**Depth and motion:** no heavy shadows; depth from 1px tonal borders and surface stepping (`#242649` → `#222B68` → `#23337C`). Hover and active use 200ms cubic-bezier(0.16, 1, 0.3, 1); card borders brighten from `#364989` to `#65A7D3` with a glow of `0 0 24px rgba(56,108,175,0.35)`. Respect reduced-motion.

**Data visualisation:** one dark-to-light ramp `#242649` → `#263B84` → `#386CAF` → `#65A7D3` → `#CDDFEF`, dashed `#364989` gridlines, mono-font ticks in `#B2D3EB`, trendlines `#65A7D3`; separate multiple series by lightness steps plus line style or markers, never new hues. Apply the same theme to all saved training and evaluation plots.

**UX:** empty, loading and error states; responsive from laptop to tablet; keyboard-accessible with visible focus rings; every metric shows its unit and evaluation context.

---

## STAGE 11 — LEAN TESTS AND SMOKE RUN (END OF PART 1)

- Tests through uv: converter correctness, box clamping, tile coordinate round-trips, merge and NMS behaviour, metric calculations on hand-made examples, config loading, API smoke tests.
- A quick regression-guard evaluation on a small fixed subset.
- Run the **full pipeline in smoke mode**, confirm the dashboard and live pipeline start against the smoke results, then commit and tag.
- **Handoff message (end of Part 1):** the exact command for Part 2, estimated duration, how to resume after interruption, and the phrase to send when finished ("pipeline finished"). Then stop.

---

## STAGE 12 — FINALISE (PART 3) AND LearnProject.md

After "pipeline finished": ingest real results, compare with the targets (state plainly what was and was not achieved), one browser verification pass of every dashboard page against the design tokens (laptop and tablet widths, contrast, focus rings, hover), one fix iteration at most, run the tests, finalise the README (setup with uv, how to run each script, the final **Model Evaluation** table in the same format as mine with new rows and hardware labelled, architecture diagram, project structure), the model card, and **LearnProject.md**. Commit, tag, and give a final summary of what improved, what did not, remaining limitations and next steps.

### LearnProject.md — specification

A single teaching document for a learner with intermediate Python and basic ML, written in plain language with analogies, **using the project's real numbers, plots and sample images** (never invented figures; anything not measured is marked "not measured"). Draft the result-independent parts in Part 1; fill every result-dependent part in Part 3. Structure:

1. **The project in two minutes:** the problem (floating plastic in rivers), the approach, the outcome in one table, and a simple pipeline diagram (Mermaid is fine).
2. **ML approach at a glance:** what object detection is, boxes, confidence, IoU, non-maximum suppression, transfer learning.
3. **Dataset:** source and size, countries, Pascal VOC versus YOLO format and how the conversion works, audit findings with figures (small-object share, boxes per image, duplicates), why the split is leakage-safe and frozen, the tiled subset and why tiling helps tiny bottles, label-quality findings. Include the contact sheet.
4. **Models used:** the AutoML MobileNet-SSD TFLite model (what it is, why it is tiny and slow here) and YOLOv8 (backbone, neck, head in plain terms, COCO-pretrained start), with a trade-off table of size, accuracy and speed, and why each was chosen.
5. **Training:** what training does, key settings in plain words (image size, epochs, batch, learning-rate schedule, freezing, augmentation and the reason for each), the losses in one paragraph, how CPU-only training was budgeted, the ablation table and what each result taught, how to read the training curves, how pause and resume works.
6. **Validation:** what the validation set is for, precision, recall, F1, mAP@50 and mAP@50-95 explained with a worked example, how the confidence threshold was chosen, how to spot overfitting.
7. **Testing:** why the test set is frozen, the evaluation protocol (thresholds, hardware, threads, subset used for slow models), the baseline-versus-new table, small/medium/large analysis, tiled versus single-pass, latency results, error analysis with real false-positive and false-negative images, robustness checks, and an honest statement of which targets were met.
8. **Inference and real time:** tiling and merging, tracking and unique counts, and the three-thread capture/inference/display design explained with the producer-consumer analogy and why stale frames are dropped.
9. **Edge deployment:** the export formats, quantization in plain words, accuracy-versus-speed results, and which model suits which device.
10. **Dashboard:** what each page shows and which file or model feeds it, how the design system is applied, how to launch it, and how results flow from the pipeline into the pages.
11. **Limitations and next steps.**
12. **Check your understanding:** eight short questions with answers, plus a glossary of every term used.
13. **How to reproduce everything** with uv, step by step.

---

**Begin Part 1 now.** Write the one-page plan, then AGENTS.md and PROGRESS.md, then build Stages 1–11 without stopping. Ask questions only if a blocker needs one.
