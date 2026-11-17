# spt-pipeline

[![Python ≥3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()
[![Version 2.0](https://img.shields.io/badge/version-2.0.0-orange)]()

A **Single-Particle Tracking (SPT)** analysis suite that takes you from raw microscopy movies (ND2 or TIFF) to publication-quality figures comparing molecular mobility across experimental conditions. It detects and links particles, measures how far they travel over defined time intervals, computes diffusion coefficients and mean-squared displacements, and runs the statistics needed to compare conditions — all driven by a single command-line interface and a plain-text manifest that records what every movie is.

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The manifest: describing your data](#the-manifest-describing-your-data)
- [Building the manifest (`prep scan`)](#building-the-manifest-prep-scan)
- [Tracking particles (`track trackpy`)](#tracking-particles-track-trackpy)
- [Measuring step sizes (`compute tau-steps`)](#measuring-step-sizes-compute-tau-steps)
- [Comparing conditions (`analyze compare-conditions`)](#comparing-conditions-analyze-compare-conditions)
- [How step sizes are summarized](#how-step-sizes-are-summarized)
- [How conditions are compared statistically](#how-conditions-are-compared-statistically)
- [Output structure](#output-structure)
- [Re-running efficiently](#re-running-efficiently)
- [Reproducibility](#reproducibility)
- [Additional tools](#additional-tools)
- [Companion programs](#companion-programs)
- [Parameter reference](#parameter-reference)
- [Tips and best practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)
- [License and citation](#license-and-citation)

## Overview

`spt-pipeline` covers the full single-particle-tracking analysis path: reading raw movies, detecting and linking particles, quantifying displacement and diffusion, and comparing experimental conditions with appropriate statistics. Work is organized so that the expensive steps (tracking) run once and are cached, while the fast steps (step-size histograms, condition comparison, statistics) can be re-run at will with different parameters.

The package installs three console scripts:

- **`spt-pipeline`** — the main analysis pipeline
- **`spt-coloc`** — two-color bead-based registration and co-diffusion analysis
- **`spt-metadata`** — a lightweight ND2/TIFF acquisition-metadata summarizer

`spt-pipeline` is organized into command groups. The four that form the core analysis path are:

- **`prep scan`** — scan your raw data and build a manifest describing every movie
- **`track trackpy`** — detect and link particles in each movie, compute per-movie diffusion/MSD, fixed-τ step sizes, and localization QC
- **`compute tau-steps`** — add displacements at further time intervals, without re-tracking
- **`analyze compare-conditions`** — pool by condition and produce comparison figures and statistics

**`workflow auto`** runs all four end to end.

The remaining groups (`prep` conversions, `track dwelltime`, `analyze brightness-mobility`, `plot`, `draw`, `workflow legacy-auto`, `utils`) provide format conversion, specialized analyses, and figure utilities, and are described under [Additional tools](#additional-tools).

## Installation

### From PyPI (recommended)

```bash
pip install spt-pipeline                # core analysis only
pip install "spt-pipeline[all]"         # all optional features (tracking + imaging)
```

Optional extras:

| Extra      | Adds                              | Used for                                     |
|------------|-----------------------------------|----------------------------------------------|
| `tracking` | `trackpy`, `numba`                | detecting/linking particles (`track trackpy`)|
| `imaging`  | `nd2`, `pims`, `tifffile`         | reading ND2/TIFF movies and metadata         |
| `stats`    | `statsmodels`                     | optional extended statistics                 |
| `all`      | everything above                  | full functionality                           |
| `dev`      | `build`, `twine`, `pytest`        | building/testing the package                 |

Measuring step sizes and comparing conditions require only the core dependencies, so re-analysis works on any machine without the tracking/imaging stack.

### From source

```bash
git clone https://github.com/andrewbazley/spt-pipeline.git
cd spt-pipeline
pip install -e ".[all]"
```

### Requirements

- Python 3.10+ (runs on macOS, Windows, and Linux, including HPC clusters)
- Installed automatically: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `scikit-image`, `seaborn`

## Quick start

The whole core path in one command:

```bash
spt-pipeline workflow auto /data/exp_05_12 /data/exp_06_03 -o results/ --mpp 0.183 --fps 50
```

This scans, tracks, measures step sizes, and compares conditions. It stops after the scan if any `condition`/`experiment` cell is still `FILL_IN` — edit `results/manifest.csv` and re-run the same command to continue.

Or run the stages yourself:

```bash
# 1. Scan raw data into a manifest, then open manifest.csv and correct any FILL_IN cells.
spt-pipeline prep scan  /data/exp_05_12  /data/exp_06_03  /data/exp_06_04 \
    --output manifest.csv --mpp 0.183 --fps 50

# 2. Detect and link particles, measure displacements at tau = 1, 5, 10 frames,
#    and write localization QC figures. Results are cached.
spt-pipeline track trackpy -m manifest.csv -o results/

# 3. (optional) Add further lags later, without re-tracking.
spt-pipeline compute tau-steps -m manifest.csv -o results/ --tau 20 40

# 4. Compare conditions: step-size distributions, diffusion, and statistics.
spt-pipeline analyze compare-conditions -m manifest.csv -o results/
```

Every command reads the manifest, checks it, and skips work that is already done, so the sequence above is safe to re-run.

### Portable manifests

Manifest paths are stored **relative to the manifest file**, using forward slashes. A manifest and its data therefore move together between Windows, macOS, and Linux with no edits. If the dataset is mounted somewhere other than beside the manifest — a different drive letter, a network share, a container mount — pass `--data-root /path/to/data` to any command.

### Movies that yield no tracks

If a movie has no particles above threshold, or no track long enough to survive the length filter, that is recorded as an outcome rather than an error: the run continues, `tracks_csv` stays blank for that row, a `{movie}_NO_TRACKS.json` records the reason and the thresholds in force, and every downstream stage skips the movie with a warning. A path that is *listed* in the manifest but missing from disk is still a hard error — that one is a real fault.

## The manifest: describing your data

The manifest is a single CSV that records everything the pipeline needs to know about your dataset: where each movie is, what condition and biological replicate it belongs to, and its acquisition parameters. It is generated for you by `prep scan` and then reviewed in any spreadsheet editor — you never fill it in from scratch. Once created, every stage refers to it, so metadata is defined in exactly one place.

| Column | Required | Auto-filled | Meaning |
|--------|:--:|:--:|---|
| `movie_id` | yes | yes | Unique identifier for the movie (from the filename) |
| `source_path` | yes | yes | Path to the raw file, stored relative to the manifest (OS-portable) |
| `file_format` | yes | yes | `nd2`, `tif`, or `tiff` (or `tracks` for pre-tracked data) |
| `condition` | yes | attempted | Experimental condition (e.g. DMSO, Tg) — verify the inference |
| `experiment` | yes | attempted | Biological replicate / imaging day, as an ISO date |
| `tracks_csv` | | filled after tracking | Path to the movie's tracks CSV |
| `diff_csv` | | filled after tracking | Path to the movie's diffusion CSV |
| `mpp` | | attempted | Microns per pixel (per-movie override of the default) |
| `fps` | | attempted | Frames per second (per-movie override of the default) |
| `n_frames`, `n_channels` | | yes | Read from the file metadata |
| `notes` | | | Free text for your own annotations |

**Terminology.** A **movie** is one field of view. An **experiment** is a biological replicate, typically one imaging day, and is the level at which conditions are compared statistically. A **condition** is an experimental treatment pooled across its experiments. Several movies may belong to the same condition and experiment.

**Validation** runs automatically at the start of every stage. It stops with a clear message if any `condition`/`experiment` is still `FILL_IN`, a referenced file is missing, `movie_id`s collide, or `mpp`/`fps` are not positive. It also warns about conditions with only one experiment or movie, and about inconsistent `mpp`/`fps` within an experiment.

## Building the manifest (`prep scan`)

```bash
spt-pipeline prep scan <dir> [<dir> ...] --output manifest.csv [options]
```

The scan walks each directory recursively, finds all `.nd2`/`.tif`/`.tiff` files (skipping hidden and macOS AppleDouble files), reads their metadata, and infers labels:

- **Condition** is taken from an existing `groupnames.csv` if present, then from name tokens (e.g. NT, Tg, DMSO, control), otherwise set to `FILL_IN`.
- **Experiment** is parsed from a date in the folder or file name (e.g. `exp_06_03`, `20260603_...`), falling back to the file modification date.

**Options**

- `--output` / `-o`: manifest path (default `manifest.csv`).
- `--mpp`, `--fps`: default acquisition values (default `0.183` µm/pixel, `50` fps), used when a movie's metadata does not supply them.
- `--exclude-pattern`: filename substring to skip (repeatable), e.g. bead or flatfield calibration movies.
- `--condition-tokens`: comma-separated tokens to match for condition inference.
- `--from-tracks`: discover existing `*_tracks.csv` files instead of raw images, so already-tracked data can be analyzed without re-tracking.

The scan writes the draft manifest plus a companion `manifest.scan_log.json` documenting every inference it made, and prints how many movies were fully resolved versus needing attention.

## Tracking particles (`track trackpy`)

```bash
spt-pipeline track trackpy -m manifest.csv -o results/ [options]
```

For each movie, the tracker loads the image (selecting a channel if the file is multichannel), detects particles with `trackpy`, links them into trajectories, and discards trajectories shorter than the minimum length. It then computes mean-squared displacements, a per-particle diffusion table, **fixed-τ displacements at τ = 1, 5, 10 frames**, and a set of **localization QC figures**, all using that movie's `mpp`/`fps` from the manifest. Movies are processed in parallel, and any movie whose outputs already exist is skipped unless `--force` is given, so an interrupted run resumes cleanly.

Each movie gets its own subdirectory under the output folder containing:

| File | Contents |
|------|----------|
| `{movie}_tracks.csv` | Linked trajectories (`particle`, `frame`, `x`, `y`, `mass`, …) |
| `{movie}_step_sizes.csv` | Displacement between consecutive detections in each track |
| `{movie}_tau_steps_{NNN}.csv` | Fixed-τ displacements, one file per lag (τ = 1, 5, 10 by default) |
| `{movie}_diffusion_coefficients.csv` | Per-particle diffusion summary (see below) |
| `{movie}_IMSD.csv`, `{movie}_EMSD.csv` | Individual and ensemble mean-squared displacement vs lag |
| `{movie}_MSD_vs_tau.csv` | Combined MSD-vs-lag summary |
| `{movie}_context.json` | The exact tracking parameters used, the lags computed, and the QC metrics |
| `qc/` | Localization QC figures (see below) |
| diagnostic plots | step histogram, log-D histogram, MSD-vs-τ curve, track overlay |

Key columns in `*_diffusion_coefficients.csv` include `particle`, `n_spots` (detections in the track), `alpha` (anomalous-diffusion exponent from the MSD log-log slope), `mean_x_px`/`mean_y_px`, `mass_per_frame` (mean brightness), and the diffusion coefficient at the chosen lag (`D_fit_um2_s`) and at the last available lag (`D_last_um2_s`).

### Localization QC (`qc/`)

Written per movie unless `--no-qc-plots`. These are the diagnostics that justify a parameter choice rather than merely record it:

| Figure | What it tells you |
|--------|-------------------|
| `{movie}_qc_annotated_frame.png` | Detections circled on frame 0 — are you finding molecules or noise? |
| `{movie}_qc_mass_histogram.png` | Detection-brightness distribution with `--minmass` drawn on, and the fraction kept |
| `{movie}_qc_subpixel_bias.png` | Fractional part of the fitted centroids. Flat is correct; peaks at 0 or 0.5 are **pixel biting** from a `--diam` that is too small, which inflates short displacements and biases D upward. A numeric deviation score and verdict are printed in the title |
| `{movie}_qc_mass_vs_size.png`, `..._mass_vs_ecc.png` | Real diffraction-limited spots form a compact cloud; tails are aggregates, debris, or merged spots |
| `{movie}_qc_features_per_frame.png` | Detections per frame, with the last/first-decile ratio — steep decline is photobleaching, a sudden step is focus loss |
| `{movie}_qc_track_length_hist.png` | Track lengths with `--min-track-length` marked, making the survivorship from the stub filter explicit |

The scalar versions of these diagnostics (`subpixel_bias_max_dev`, `bleach_ratio_last_first_decile`, `frac_detections_above_minmass`, …) are also stored in `{movie}_context.json` so they can be compared across movies without opening figures.

**Options**

`--mpp` (0.183), `--fps` (50), `--diam` (7), `--minmass` (40), `--range` (5, linking search radius in pixels), `--memory` (0, frames a particle may vanish), `--min-separation` (3), `--percentile` (90), `--min-track-length` (4), `--diffusion-tau` (5, the lag at which `D_fit` is computed), `--channel` (0), `--tau` (`1 5 10`), `--no-tau-steps`, `--no-qc-plots`, `--n-jobs` (8), `--force`, `--data-root`.

Brightness/flatfield correction is **off by default**; pass `--correct-brightness` to enable a 2D-polynomial illumination correction of per-track brightness.

## Measuring step sizes (`compute tau-steps`)

Step sizes at τ = 1, 5, 10 frames are computed automatically by `track trackpy`. This command exists to **add further lags later without re-tracking**:

```bash
spt-pipeline compute tau-steps -m manifest.csv -o results/ [--tau 1 3 5 8] [options]
```

A **step size** is the distance a particle travels over a chosen time interval, τ. For each requested τ (in frames), this step pairs every detection at frame `f` with the detection at frame `f + τ` in the same track, whenever both exist, and records the displacement. This sliding-window measurement is the standard way to build a step-size distribution: a long track contributes many overlapping displacements at a given τ. Because diffusivity in cytoplasm is scale-dependent, step sizes and diffusion are always reported at a specific τ.

For each movie and τ, the tracks file is read once and a CSV `{movie}_tau_steps_{NNN}.csv` is written next to the movie's other outputs, with columns `particle`, `frame_a`, `frame_b`, `tau_frames`, `tau_s`, `dx_px`, `dy_px`, `step_px`, `step_um`. Existing per-τ files are skipped unless `--force`, so adding a new τ only computes that τ.

**Options**

- `--tau`: one or more lags in frames (default `1 3 5 8`).
- `--mpp`, `--fps`: defaults when a movie does not override them.
- `--min-nspots`, `--min-alpha`: restrict to tracks passing these thresholds (read from the diffusion table).
- `--non-overlapping`: use a τ-spaced grid of start frames, giving statistically independent displacements at reduced count (the default sliding window is recommended for distributions).
- `--n-jobs` (8), `--force`.

## Comparing conditions (`analyze compare-conditions`)

```bash
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ [options]
```

This step pools the cached step-size and diffusion tables by condition and produces the comparison figures and statistics. It reads only the small cached CSVs (never the large tracks files), so it runs in seconds and can be repeated with different bin widths, filters, or groupings. If `--tau` is omitted, it automatically uses whichever lags were computed.

For each τ it produces a step-size distribution comparing conditions (see [below](#how-step-sizes-are-summarized)), a cross-τ overlay per condition, and a per-unit summary. Every step-size figure shows τ in both frames and seconds, shares identical axes across conditions and representations (scaled to the full data so no tail is clipped), and is annotated with statistics at the automatically chosen unit (see [below](#how-conditions-are-compared-statistically)). It also produces diffusion-coefficient distributions (linear and log scale) and `D_boxplot_by_condition.png` — box = IQR and whiskers = 10th–90th percentile of the per-movie median D, overlaying each movie's median as a dot and each experiment's median as a bold diamond, colored by biological replicate — plus MSD-vs-τ curves overlaid by condition with SEM bands. Between-condition statistics go to `statistical_tests.csv`; track-level distributional tests go to `track_level_ks_tests.csv`.

**Options**

- `--tau`: lags to include (default: all cached lags).
- `--mpp`, `--fps`, `--data-root`.
- `--min-nspots`, `--min-alpha`: track-quality filters applied at comparison time.
- `--subtract-loc-error`, `--loc-error-alpha-max` (0.3), `--loc-error-min-nspots` (10): estimate a localization-error floor from quasi-immobile particles and subtract it from diffusion coefficients.
- `--confinement-threshold-um`: report the fraction of tracks below this median step size.
- `--track-length-bins`: custom track-length strata for the length-split histogram.
- `--step-bin-width-um`, `--diffusion-bin-width`, `--logd-bin-width`: histogram bin widths (`auto`, or a number; step widths are floored to the localization precision, `mpp/10`).
- `--gmm`: **off by default.** Enable Gaussian-mixture decomposition of the diffusion distribution into subpopulations (BIC-selected component count).
- `--brightness`: **off by default.** Enable brightness–mobility correlation plots.
- `--max-gmm-k` (5).
- `--advanced`, `--advanced-only`: see below.

### Advanced trajectory analyses (`--advanced`)

Six analyses that go beyond step sizes and D, run per condition and then compared:

| Keyword | Analysis | Aliases |
|---------|----------|---------|
| `vacf` | Velocity autocorrelation, multi-δ | `velocity` |
| `cfbm` | Fractional-Brownian-motion fit to the collapsed VACF (α′, Hurst H) | `fbm`, `hurst`, `collapse` |
| `angles` | Turning-angle distributions, ⟨cos θ⟩ vs τ, KS tests | `angle`, `cos_theta`, `turning` |
| `eb` | Ergodicity-breaking parameter, Var(MSD)/⟨MSD⟩² | `ergodicity` |
| `ngp` | Non-Gaussian parameter α₂ | `a2`, `alpha2`, `non-gaussian` |
| `rheology` | Complex shear modulus G′/G″ via Mason/GSER | `rheo`, `microrheology`, `gser` |

```bash
# everything
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ --advanced

# just the VACF and alpha-2 plots
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ --advanced-only VACF a2
```

Keywords are case-insensitive; an unrecognized one is an error rather than a silent omission. `--advanced-only` implies `--advanced`. Conditions come from the **manifest**, not from filenames, so grouping matches every other stage. Track coordinates are converted to microns per movie using that movie's own `mpp`. Output lands in `compare_conditions/advanced_analysis/`: one subdirectory per condition, plus `comparison_*.csv` tables, overlay figures, and `condition_grouping.csv` recording exactly which movies went where.

Additional options: `--advanced-min-track-len` (10), `--advanced-max-tlag` (10), `--advanced-particle-radius` (0.020 µm, rheology only), `--advanced-temperature` (298.15 K), `--advanced-n-jobs` (4).

## How step sizes are summarized

For each condition and τ, the pipeline reports the step-size distribution in four complementary ways. This is because a pooled distribution weights long tracks more heavily (they contribute more displacements), and it is important to know whether that weighting is shaping the result.

| Representation | What it shows | Track-length weighting |
|---|---|---|
| **Raw pooled** (primary) | Every displacement, pooled across all tracks | Long tracks weigh more |
| **Track-weighted** | Each track's own distribution, averaged (± SEM) | Every track counts equally |
| **Track-mean** | One mean step per track | Every track counts equally |
| **Track-length-split** | Raw pooled, split by track length | Makes any length dependence visible |

When the raw-pooled and track-weighted distributions agree, track length is not distorting the comparison. When they differ, both should be reported. The pipeline characterizes this effect rather than silently "correcting" it, because longer tracks can be biologically meaningful (slower or confined particles often persist longer). A companion `cross_tau_summary.csv` records, per condition and τ, how many tracks contribute and what fraction are excluded at larger τ.

## How conditions are compared statistically

Measurements are made per track, but inference is done at the level of the biological replicate, and conditions are the unit of comparison.

### The design is chosen from your manifest

That hierarchy only holds when the dataset actually contains replicate experiments. Running a single imaging day through an experiment-level test gives every condition n = 1, which produces p-values that are undefined or meaningless — and printed on a figure, worse than nothing, because they look authoritative.

The pipeline therefore reads the manifest's `experiment` column and picks the design itself:

| Condition of the data | Design | Statistical unit | What a significant result means |
|---|---|---|---|
| Every condition has ≥ 2 experiments | **between-experiment** | experiment (biological replicate) | The effect reproduces across imaging days |
| Any condition has 1 experiment | **within-experiment** | movie (technical replicate) | The conditions differ *within this one experiment*; reproducibility is not addressed |

The gate is the *minimum* number of experiments across conditions, not the total: two experiments split 2-and-1 still leaves one arm at n = 1, so that also falls back to within-experiment. The decision, and its reason, are recorded in `compare_conditions/statistical_design.json`, stamped into every row of `statistical_tests.csv` (`unit`, `design`, `interpretation`), and printed on every annotated figure — a within-experiment run carries an explicit "⚠ single experiment — technical replicates, not biological" line.

### The tests

- Per-unit summaries (median step, median D, MSD at the fit lag) are computed for each condition–unit pair. Both `per_{unit}_summary.csv` and `per_experiment_summary.csv` are written, so day-to-day variation stays inspectable even when the movie is the unit.
- **Mann–Whitney U** compares those per-unit summaries between conditions (nonparametric, appropriate for small n).
- **Cliff's delta** reports effect size, and **Benjamini–Hochberg** correction is applied when more than two conditions are compared.
- **Bootstrap confidence intervals** on condition medians resample at the statistical unit.
- **Track-level tests** (KS, Mann–Whitney, Cliff's delta on the pooled per-track values) are always computed and written to `track_level_ks_tests.csv` — and deliberately never annotated on a figure. There, n is the number of tracks, so essentially any difference reaches significance; those p-values describe the two *distributions*, not whether the effect replicates. Read `cliffs_delta`, which does not inflate with n.

Individual steps are never treated as independent replicates, and individual movies are treated as the unit only when the data leave no alternative — with the label to say so.

## Output structure

```
results/
├── {movie}/                         # one folder per movie (tracks, MSD, diffusion, tau-steps, plots)
│   ├── {movie}_NO_TRACKS.json       # only when the movie yielded nothing (skipped downstream)
│   └── qc/                          # localization QC panels
├── run_logs/                        # one JSON per command invocation
└── compare_conditions/
    ├── run_parameters.json
    ├── histogram_bin_settings.json
    ├── statistical_design.json      # which unit was chosen, and why
    ├── statistical_tests.csv        # unit-level tests (annotated on the figures)
    ├── track_level_ks_tests.csv     # pooled-track tests (never on the figures)
    ├── condition_summary.csv
    ├── step_size_analysis/
    │   ├── tau_001/  (raw_pooled / track_weighted / track_mean / track_length_split histograms
    │   │             as .png + _bins.csv, plus per_{unit}_summary.csv and
    │   │             per_experiment_summary.csv)
    │   ├── tau_005/  tau_010/  …
    │   └── cross_tau/  (per-condition multi-τ overlays, cross_tau_summary.csv)
    ├── diffusion_analysis/          # D histograms (linear/log), D_boxplot_by_condition.png,
    │                                # D_boxplot_*_medians[_wide].csv, per-unit summaries
    │                                # (+ GMM subpopulations only with --gmm)
    ├── msd_analysis/                # MSD-vs-τ overlaid by condition (linear and log-log)
    ├── advanced_analysis/           # only present with --advanced / --advanced-only
    └── brightness_analysis/         # only present with --brightness
```

All figures use a consistent, colorblind-safe palette (Okabe–Ito for conditions; viridis/cividis for continuous values) at 300 dpi.

## Re-running efficiently

The separation between tracking, step measurement, and comparison keeps re-analysis fast — for example when reviewing at different τ values:

| Change | What re-runs | Typical time (~57 movies) |
|---|---|---|
| First analysis from raw data | track → tau-steps → compare | 30–60 min (tracking dominates) |
| Add a new τ | tau-steps for that τ → compare | 2–5 min |
| Change bin widths, filters, or grouping | compare only | 30–60 s |
| Add a new imaging day | track (new movies) → tau-steps → compare | 10–20 min |

## Reproducibility

Every command writes a JSON log to `results/run_logs/` recording the full set of parameters, the manifest path and its SHA-256 hash, the package version, timing, per-movie status, the files produced, and library versions. Histogram bin choices are saved in `histogram_bin_settings.json`, and each movie's tracking parameters in its `{movie}_context.json`. Given a run log, a figure can be reproduced exactly.

## Additional tools

These commands are preserved from the established pipeline and cover format conversion, specialized analyses, and figure generation. Dwell-time and brightness analyses are separate, opt-in commands and do not run as part of the core path.

### Data preparation

- **`prep trackmate <root>`** — convert TrackMate CSV exports to a standardized trajectory format, inferring condition/replicate from the directory structure.
- **`prep nd2-to-tiff <dir>`** — extract a channel from each ND2 and write TIFFs.
- **`prep stack-channels <spool> <out>`** — combine `lefts/` and `rights/` TIFF slices into a two-channel OME-TIFF.
- **`prep find-ome-tif <dir>`** — list Micro-Manager OME-TIFF stacks found under a directory.

### Specialized analyses

- **`track dwelltime <dir>`** — detect and link particles and compute how long each particle remains visible; writes per-movie filtered tracks and dwell diagnostics.
- **`analyze brightness-mobility <input>`** — relate per-track brightness to mobility (step size, immobile fraction, uncorrected/corrected D), with scatter, histogram, violin, and CDF plots.

### Plotting and rendering

- **`plot steps-brightness <dir>`** — 2D heatmaps of step size vs brightness and step-size-colored track overlays.
- **`plot dwell <dir>`** — density-normalized dwell-time histograms by condition.
- **`plot brightness-heatmap <dir>`** — condition-pooled brightness–step heatmaps.
- **`plot brightness-map <dir>`** — brightness-colored track overlays.
- **`draw movie-tracks <dir>`** — render movie frames with per-particle track tails for figures.

### Utilities and workflows

- **`utils split-metrics <csv>`** — split a master metrics CSV into one file per parameter, with conditions as columns.
- **`workflow legacy-auto <dir>`** — the original convenience chain for single-directory processing (brightness-mobility → steps-brightness → dwell). `workflow auto` now names the v2 core path; see [Quick start](#quick-start).

## Companion programs

### Two-color co-diffusion (`spt-coloc`)

Bead-based channel registration and two-color co-diffusion analysis, consuming track CSVs produced by `track trackpy`.

```bash
spt-coloc register beads.tif -o registration/
spt-coloc pair ch1/ ch2/ registration/transform.json --mpp 0.122 --fps 50 -o coloc/
spt-coloc analyze coloc/ --n-shuffles 500 -o coloc/
```

### Movie metadata summarizer (`spt-metadata`)

Recursively scans a directory for ND2/TIF/TIFF movies and writes one CSV row per movie with acquisition fields (frame count, exposure, pixel size, objective, lasers, …).

```bash
spt-metadata <input_directory> [--output FILE]
```

## Parameter reference

### Tracking

| Parameter | Units | Typical | Description |
|-----------|-------|---------|-------------|
| `--diam` | pixels | 5–15 | Particle diameter for detection; should match the real spot size |
| `--minmass` | intensity | 50–500 | Minimum integrated intensity to accept a detection |
| `--percentile` | 0–100 | 0–100 | Intensity percentile threshold for detection |
| `--range` | pixels | 5–20 | Maximum movement between frames allowed when linking |
| `--memory` | frames | 0–5 | Frames a particle may disappear and still be re-linked |
| `--min-track-length` | frames | 3–50 | Discard tracks shorter than this |
| `--mpp` | µm/pixel | 0.05–1.0 | Spatial calibration (default 0.183) |
| `--fps` | frames/s | 1–100 | Frame rate (default 50) |

### Step measurement and comparison

| Parameter | Units | Typical | Description |
|-----------|-------|---------|-------------|
| `--tau` | frames | 1–20 | Time interval(s) at which displacements are measured (default 1 5 10, computed during tracking) |
| `--min-nspots` | frames | 5–50 | Minimum track length to include |
| `--min-alpha` | unitless | 0–2 | Minimum anomalous exponent to include |
| `--subtract-loc-error` | flag | — | Subtract a localization-error floor from D |
| `--step-bin-width-um` | µm | auto | Step-size histogram bin width (floored to mpp/10) |
| `--n-jobs` | count | 1–N | Parallel worker processes (default 8) |

## Tips and best practices

1. **Review the manifest first.** The scan does the heavy lifting, but always check condition/experiment assignments before tracking — everything downstream depends on them.
2. **Tune detection on one movie.** Too many false positives: raise `--minmass` or `--percentile`. Broken tracks: raise `--range` or `--memory`. Missed particles: lower `--diam` or `--minmass`.
3. **Iterate cheaply.** Re-run `compute tau-steps` to add a lag, or `analyze compare-conditions` to change bins or filters — neither re-tracks.
4. **Set per-movie calibration when it varies.** Use the `mpp`/`fps` columns in the manifest to override the defaults for individual movies.
5. **Use `--n-jobs` on multi-core machines and clusters.** Movies are independent and parallelize well.

## Troubleshooting

**Validation error about `FILL_IN` values** — open the manifest and set the flagged `condition`/`experiment` cells.

**`trackpy`/`nd2` required** — install the extras: `pip install "spt-pipeline[tracking,imaging]"`.

**No particles detected** — lower `--minmass` or `--percentile`, and confirm `--diam` matches the spot size. Check `qc/{movie}_qc_mass_histogram.png` to see where the cutoff actually falls relative to the brightness distribution.

**A movie reports "no tracks"** — this is not an error; the run continues and the movie is skipped downstream. Read `{movie}_NO_TRACKS.json` for the reason and the thresholds in force, then re-run that movie with relaxed settings and `--force`.

**D looks too high / short steps look inflated** — check `qc/{movie}_qc_subpixel_bias.png`. Peaks at 0 or 0.5 in the fractional-centroid histogram mean the centroid is snapping to pixel centers; raise `--diam`.

**Tracks too short or broken** — raise `--range` to allow larger movements and `--memory` to bridge gaps.

**Nothing to compare / empty figures** — confirm `*_tau_steps_*.csv` files exist beside each movie's tracks (they are written by `track trackpy`), and that the manifest's `tracks_csv` paths resolve. If the manifest came from another machine, pass `--data-root` pointing at where the data is now mounted.

**Statistics say "technical replicates"** — the manifest has a single experiment for at least one condition, so the movie is the statistical unit. Assign real imaging days to the `experiment` column to get biological-replicate inference; see [How conditions are compared statistically](#how-conditions-are-compared-statistically).

## License and citation

MIT License — see [LICENSE](LICENSE). If this pipeline supports a publication, please cite it and the underlying [`trackpy`](http://soft-matter.github.io/trackpy/) and `scikit-learn` libraries. Step-size and diffusion-at-τ conventions follow established Holt-lab SPT analyses.
