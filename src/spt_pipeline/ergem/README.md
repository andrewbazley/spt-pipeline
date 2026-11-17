## FICD expression v3.1 (adaptive-track-territory)

The expression measurement is track-seeded outward growth (not red-channel
segmentation): trajectories with >5 steps seed the cell territory, the marker is
grown from `background + growth_z*sigma`, a global histogram background peak is
subtracted, and marker-negative fields get a valid `no_signal` value rather than
a failure. Details in `EXPRESSION_v2.md`, `CHANGELOG.md`, `VALIDATION.md`.
The canonical expression column is now `ficd_bgsub_intensity` (`ficd_bgsub_mean` kept as an alias). `crosscorr` additionally writes per-cell outputs
(`Dcell_vs_ficd_expression.csv`, `correlation_by_*_cells.csv`) that aggregate D
within each `(movie_id, cell_id)` before testing, avoiding pseudoreplication.

### Re-run: skip tracking, rewrite only the FICD files

Tracking is cached (skip-if-exists), so re-running the full command recomputes
nothing in ingest and regenerates every FICD/cross-correlation/figure file:

```bash
python -m spt_pipeline.ergem run --condition-map "$JSON" -o "$OUT" \
    --mpp 0.11 --fps 100 --tau-ms 100 --with-expression --mixed --n-jobs 8
```

Or redo just the expression stages against the existing manifest:

```bash
python -m spt_pipeline.ergem expression -m "$OUT/manifest.csv" --condition-map "$JSON" -o "$OUT" --n-jobs 8
python -m spt_pipeline.ergem crosscorr  -m "$OUT/manifest.csv" \
    --expression "$OUT/ficd_expression/trajectory_expression_pooled.csv" -o "$OUT" --n-jobs 8
python -m spt_pipeline.ergem figures    -m "$OUT/manifest.csv" -o "$OUT" --tau-ms 100 \
    --crosscorr "$OUT/cross_correlation/D_vs_ficd_expression.csv"
```

Measurement parameters pass through after `--` (e.g. `-- --growth-z 1.75
--track-bridge-radius-um 3.0`). The expression stage needs `nd2`,
`scikit-image`, `tifffile`, `matplotlib` (see `requirements.txt`).

# `spt_pipeline.ergem` — FICD complementation ER-GEM workflow

A FICD-specific layer on top of the core `spt_pipeline`. It reads the LUMEN
organisational output (`*_condition_map.json`), locates the files to analyse,
computes ER-GEM diffusion from the existing MOSAIC trajectories, measures FICD
expression, and runs the factored WT-vs-E234G × DMSO-vs-Tg comparison — the test
of whether inactivating BiP (E234G) blunts the Tg-induced rise in ER-GEM
diffusion.

Nothing here changes core behaviour. The only in-place core edits are
backwards-compatible general improvements (see *Core changes* below); the
trackpy path and all existing outputs are untouched.

## Commands

```bash
# 1) JSON -> manifest (direct reader; calibration defaults 0.11 um/px, 100 fps,
#    but ND2/TIF metadata is preferred when present)
python -m spt_pipeline.ergem from-json FICD_ERGEM_condition_map.json -o manifest.csv

# 2) MOSAIC Traj_*.csv -> per-movie diffusion/MSD products (no re-tracking)
python -m spt_pipeline.ergem ingest -m manifest.csv -o results --mpp 0.11 --fps 100 --tau-ms 100 --n-jobs 8

# 3) (optional) per-cell FICD marker measurement, driven by the map's explicit
#    red/trajectory pairs (robust to the _RED_ mid-string naming)
python -m spt_pipeline.ergem expression -m manifest.csv \
    --condition-map FICD_ERGEM_condition_map.json -o results

# 4) diffusion vs FICD expression cross-correlation table
python -m spt_pipeline.ergem crosscorr -m manifest.csv \
    --expression results/ficd_expression/trajectory_expression_pooled.csv -o results

# 5) factored contrasts (pooled + clone-resolved + optional mixed-effects)
python -m spt_pipeline.ergem contrasts -m manifest.csv -o results --mixed

# 6) ER-GEM figures (alpha-vs-logD joint, D violins, D-vs-FICD), tau = 100 ms
python -m spt_pipeline.ergem figures -m manifest.csv -o results --tau-ms 100 \
    --crosscorr results/cross_correlation/D_vs_ficd_expression.csv

# — or everything except step-size figures, in one shot —
python -m spt_pipeline.ergem run --condition-map FICD_ERGEM_condition_map.json \
    -o results --with-expression --mixed --n-jobs 8
```

Step-size histograms and the standard comparison figures come from the core
pipeline, unchanged, using the same manifest:

```bash
spt-pipeline compute tau-steps -m results/manifest.csv -o results --mpp 0.11 --fps 100
spt-pipeline analyze compare-conditions -m results/manifest.csv -o results --mpp 0.11 --fps 100
```

## What each module does

- **`condition_map.py`** — flattens `conditions[] → replicates[] → movies[]` into
  the core manifest (one row per movie) plus additive factor columns
  (`genotype, complementation, treatment_code, clone, has_paired_red,
  ficd_expr_table, condition_id, replicate_id`). `condition = condition_id`,
  `experiment = experiment_date`, `tracks_csv = trajectory_csv`. Calibration is
  resolved per movie: explicit JSON field → ND2/TIF metadata → default
  (0.11 µm/px, 100 fps) → blank (core `validate` then refuses to run).
- **`ingest.py`** — reads each MOSAIC `Traj_*.csv`, normalises columns
  (`Trajectory→particle`, `Frame→frame`), computes IMSD/EMSD (pure numpy,
  trackpy-compatible overlapping estimator), and reuses core `diffusion` to emit
  the same per-movie products the trackpy path produces
  (`*_tracks.csv`, `*_diffusion_coefficients.csv`, `*_MSD_vs_tau.csv`, …). No
  detection/linking, no trackpy dependency.
- **`expression.py`** — stages `Red_<movie_id>.nd2` / `Traj_<movie_id>.csv`
  symlinks from the map's explicit pairs so the vendored
  `_ficd_cell_intensity.py` always aligns red↔trajectory (its own `image_key`
  matcher assumes a *leading* `Red_` prefix, which the `..._RED_001.nd2` files do
  not have). Pooled `trajectory_expression.csv` rows are keyed back to
  `movie_id`.
- **`crosscorr.py`** — joins per-track D to per-track FICD (`ficd_bgsub_mean`) on
  `(movie_id, particle)` and writes `D_vs_ficd_expression.csv` plus Spearman
  summaries per condition and per (condition, experiment). Marker-negative WT and
  the no-red top-up movies are excluded from correlation (kept for diffusion).
- **`figures.py`** — three figures paneled by genotype group (FICD-WT / FICD-E234G / WT-untransfected), each comparing DMSO vs Tg: alpha vs log10(D at tau) joint plot (marginal KDEs, per-treatment median guides, 2D contours), a D box/whisker (box=IQR, whiskers=10-90 pct, no shading) with per-movie points colour-coded by replicate, per-replicate median diamonds and DMSO-vs-Tg significance, and a D-vs-FICD joint plot (linear FICD x, log10 D y) in the same style as the alpha plot.
- **`contrasts.py`** — per-experiment-median Mann-Whitney (+ Cliff's δ, BH within
  family) for: WT vs E234G within each treatment; Tg vs DMSO within each
  complementation; and the interaction (per-experiment Tg−DMSO Δ, WT vs E234G).
  Clone-resolved variants flag clones with N<3 (`low_power`). Optional statsmodels
  mixed model: `log10 D ~ complementation*treatment`, random experiment intercept
  and clone variance component.

## Notes for this dataset

- With N=3 biological replicates the smallest two-sided Mann-Whitney p is 0.1; the
  real map has N=5–7 per arm, which does better. Hold conclusions to the
  experiment-level tests, not pooled-track counts.
- **C11** is N=2 in every cell — fine for pooling, flagged `low_power` in
  clone-resolved output; no clone-resolved statistics are trustworthy for it.
- The 22 A8/E234G/Tg movies from 20240625 (no red channel) count as diffusion
  replicates but carry no expression readout, so they drop from cross-correlation.

### D at tau = 100 ms

`ingest --tau-ms 100` sets the D fit lag to the frame nearest 100 ms (10 frames at 100 fps), so `D_fit_um2_s` is D_100ms and every figure/contrast/cross-correlation uses it. Pass `--tau-fit <frames>` to override, or `--tau-ms` with a different value.

## Performance (large datasets)

Every stage parallelises and reports progress; default `--n-jobs 8` (capped at
CPU count) on `ingest`, `expression`, `crosscorr`, `contrasts`, `figures`, `run`.

- `ingest` runs across a process pool with a tqdm bar (movie X/N, rate, ETA;
  plain-text fallback if tqdm is absent). Skip-if-exists makes it resumable.
- IMSD is computed vectorised and in memory only; the manifest is written once
  (not per movie).
- `expression` runs experiments concurrently; `crosscorr`/`contrasts`/`figures`
  read the per-movie `diff_csv` files across a thread pool.

### Computed vs skipped (only what figures/outputs use)

`ingest` computes and writes, per movie: trimmed `*_tracks.csv`,
`*_diffusion_coefficients.csv` (alpha + D at the fit lag), `*_MSD_vs_tau.csv`
(from IMSD), `*_context.json`. IMSD is the in-memory source of alpha and D.
Deliberately **not** produced (unused, so skipped for speed): EMSD, the
`*_IMSD.csv` / `*_EMSD.csv` / `*_step_sizes.csv` files, and dwell_time/total_mass
columns. `ingest` prints this note at the start of each run.

## Core changes (in place, backwards-compatible)

- `utils.canonicalize_track_columns` (called inside `normalize_track_table`):
  aliases tracker-specific column names to `particle/frame/x/y/mass`. Exact then
  case-insensitive matching only (never substring), so trackpy tables pass
  through untouched. This is what lets the core read MOSAIC CSVs.
- `manifest._extract_image_metadata`: also reads µm/px and fps from TIF
  (ImageJ/OME/resolution tags) in addition to ND2. Only fills when found.

Core CLI defaults remain 0.183 µm/px / 50 Hz (the BiP-SPT values); the ER-GEM
default (0.11 / 100) lives only in this subpackage.
