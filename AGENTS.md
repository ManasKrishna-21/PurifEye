# AGENTS.md — PurifEye Operating Rules

## Core Principles
1. **Tooling:** `uv` only (`uv run`, `uv add`, `uv sync`). Never raw `pip`.
2. **Execution Model:**
   - **Part 1 (Build & Smoke Test):** Stages 1–11 completed in one go with concise outputs. Stop at handoff command.
   - **Part 2 (Unattended Run):** User executes the one-command pipeline on their terminal with zero agent credits.
   - **Part 3 (Finalize):** Model evaluation ingestion, browser visual audit, finalized documentation, commit & tag.
3. **Credit Discipline:**
   - Single-line progress statements between steps.
   - Summaries under 5 lines. No large log dumps or echoing source files.
   - Complete single-pass file writes. Test only on tiny subsets (50 images / 1 epoch).
   - No polling on long jobs.
4. **Safety & Integrity:**
   - Do not modify or move raw dataset in `Dataset/` or existing baseline models.
   - All intermediate and final run artifacts belong in `runs/`, `yolo_dataset/`, or `reports/`.
   - Never invent or fabricate metrics; state hardware, split, and thresholds beside every metric.
5. **Concurrency & Code Style:**
   - Standard Python threads and Queues (`threading`, `queue.Queue`). Never `async/await`.
   - Idempotent and resumable scripts with step completion markers.
6. **Design System Tokens (Blue Palette Only):**
   - Canvas: `#242649` | Cards: `#222B68` | Elevated: `#23337C`
   - Borders: `#364989` | Primary Action: `#386CAF` | Pressed: `#263B84`
   - Accent / Bounding Boxes / Reticle: `#65A7D3` (with `#242649` halo)
   - Text: `#FFFFFF` (Primary), `#B2D3EB` (Secondary), `#CDDFEF` (Headers/Emphasized)
