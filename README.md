# spt-pipeline

[![Python ≥3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()
[![Version 2.1](https://img.shields.io/badge/version-2.1.0-orange)]()

Takes single-particle-tracking movies and gives you back figures comparing how
molecules move between conditions. It finds the particles, links them into
tracks, measures how far they travel, works out diffusion coefficients, and runs
the statistics — driven from the command line by one plain CSV that says what
each movie is.

- [Install](#install)
- [Quick start](#quick-start)
- [The manifest](#the-manifest)
- [Building the manifest: `prep scan`](#building-the-manifest-prep-scan)
- [Tracking: `track trackpy`](#tracking-track-trackpy)
- [Adding more lags: `compute tau-steps`](#adding-more-lags-compute-tau-steps)
- [Expression markers: `analyze marker-intensity`](#expression-markers-analyze-marker-intensity)
- [Comparing conditions: `analyze compare-conditions`](#comparing-conditions-analyze-compare-conditions)
- [Filtering tracks](#filtering-tracks)
- [How step sizes are summarized](#how-step-sizes-are-summarized)
- [How conditions are compared](#how-conditions-are-compared)
- [Advanced trajectory analyses](#advanced-trajectory-analyses)
- [Output structure](#output-structure)
- [Re-running efficiently](#re-running-efficiently)
- [Reproducibility](#reproducibility)
- [Companion tools](#companion-tools)
- [Parameter reference](#parameter-reference)
- [Troubleshooting](#troubleshooting)
- [License and citation](#license-and-citation)

## Install

```bash
pip install spt-pipeline            # re-analyse existing tracking output
pip install "spt-pipeline[all]"     # everything, including tracking and image reading
```

The core install is deliberately small — numpy, pandas, matplotlib, scipy — so a
collaborator can reproduce your figures from cached CSVs without installing
trackpy or the ND2 reader. Add extras only for what you actually do:

| Extra | Adds | Needed for |
|---|---|---|
| `tracking` | trackpy, numba | finding and linking particles |
| `imaging` | nd2, pims, tifffile | reading ND2/TIFF movies and marker images |
| `gmm` | scikit-learn | `compare-conditions --gmm` |
| `coloc` | scikit-image | `spt-coloc` bead registration |
| `stats` | statsmodels | mixed-effects contrasts in `ergem` |

From source:

```bash
git clone https://github.com/andrewbazley/spt-pipeline.git
cd spt-pipeline && pip install -e ".[all]"
```

Python 3.10+, on macOS, Windows and Linux.

## Quick start

Everything in one command:

```bash
spt-pipeline workflow auto /data/2026-08-05 -o results/ --mpp 0.11 --fps 100
```

That scans your data, tracks every movie, measures step sizes, and compares
conditions. It stops after the scan if it couldn't work out a condition or a
date for some movie, so you can fix the manifest before spending an hour of
compute on mislabelled data. Fix it, run the same command again, and it picks up
where it left off.

Or run the stages yourself:

```bash
# 1. Look at the data and write a manifest. Open it and fix any FILL_IN cells.
spt-pipeline prep scan /data/2026-08-05 --output manifest.csv --mpp 0.11 --fps 100

# 2. Find and link particles. Also computes step sizes at tau = 1, 5, 10 frames
#    and writes QC figures. This is the slow step, and it's cached.
spt-pipeline track trackpy -m manifest.csv -o results/

# 3. Compare conditions. Reads only cached CSVs, so it takes seconds.
spt-pipeline analyze compare-conditions -m manifest.csv -o results/
```

Every command reads the manifest, checks it, and skips work that's already done,
so re-running is always safe.

## The manifest

One CSV describing your dataset. `prep scan` writes a draft; you open it in a
spreadsheet and fix anything it got wrong. After that, every stage reads it, so
your metadata lives in exactly one place.

| Column | Auto-filled | What it is |
|---|:--:|---|
| `movie_id` | yes | Unique name, from the filename |
| `source_path` | yes | Where the movie is, relative to the manifest |
| `file_format` | yes | `nd2`, `tif`, `tiff`, or `tracks` |
| `condition` | attempted | What was done to the cells — check this |
| `experiment` | attempted | Biological replicate / imaging day, as `YYYY-MM-DD` |
| `tracks_csv` | by tracking | Where the tracks ended up |
| `diff_csv` | by tracking | Where the diffusion table ended up |
| `marker_image` | by marker stage | The paired expression-reporter image |
| `mpp`, `fps` | from metadata | Microns per pixel, frames per second |
| `n_frames`, `n_channels` | from metadata | Used to spot single-frame files |
| `notes` | — | Free text. `marker` here marks an expression image |

Anything it couldn't infer is written as `FILL_IN`, and the pipeline refuses to
run until you've fixed it.

### Manifests move between computers

Paths are stored relative to the manifest, with forward slashes. Move the folder
to another machine — Windows to Mac, laptop to cluster — and it just works.

Reading is more forgiving than writing. If a stored path doesn't resolve,
because the manifest is old or the share is mounted somewhere new, the loader
looks for the deepest matching tail of that path under the manifest's own
directory. A path like `\\server\share\…\20260805 Expt\C11-157_DMSO\488__001.nd2`
is found at `C11-157_DMSO/488__001.nd2` no matter what sat above it before. The
longest match wins, and a bare filename is never matched by searching, so it
can't quietly grab the wrong file.

If your data lives somewhere other than beside the manifest, point at it:

```bash
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ \
    --data-root /Volumes/lab-share/2026-08-05
```

### Movies that produce nothing

If a movie has no particles bright enough, or no track long enough to keep,
that's a result, not a crash. The run continues, `tracks_csv` stays empty for
that row, a `{movie}_NO_TRACKS.json` records why and what the thresholds were,
and every later stage skips it with a note. A path that *is* listed but missing
from disk is still a hard error — that one's a real problem.

Marker images and any single-frame file are never handed to the tracker. There's
no time axis to link along.

## Building the manifest: `prep scan`

```bash
spt-pipeline prep scan <dir> [<dir> ...] --output manifest.csv [options]
```

Walks the directories, finds ND2 and TIFF files, reads frame count, channel
count, pixel size and frame rate from the file headers, and guesses condition
and experiment.

- **Condition** comes from a `groupnames.csv` if there is one, otherwise from
  matching tokens in the filename or folder name (`NT`, `Tg`, `DMSO`, `control`,
  and so on — override with `--condition-tokens`).
- **Experiment** comes from a date in the folder or filename, and falls back to
  the file's modification time, which is flagged for you to check.

Every guess is written to a `.scan_log.json` beside the manifest, so you can see
what it decided and why.

**Options:** `--mpp`, `--fps` (fallbacks when the file doesn't say),
`--exclude-pattern` (repeatable, skips beads and flatfields),
`--condition-tokens`, `--from-tracks` (build a manifest from existing
`*_tracks.csv` instead of raw movies), `--data-root`.

## Tracking: `track trackpy`

```bash
spt-pipeline track trackpy -m manifest.csv -o results/ [options]
```

For each movie: load it, pick a channel if it's multichannel, find particles
with trackpy, link them into tracks, throw away tracks shorter than the minimum,
then compute mean-squared displacements, a per-particle diffusion table,
fixed-tau step sizes, and QC figures. Movies run in parallel, and any movie
that's already done is skipped unless you pass `--force`.

Each movie gets its own folder:

| File | What's in it |
|---|---|
| `{movie}_tracks.csv` | Linked trajectories: `particle`, `frame`, `x`, `y`, `mass`, … |
| `{movie}_tau_steps_{NNN}.csv` | Step sizes at one lag; one file per lag |
| `{movie}_step_sizes.csv` | Consecutive-frame displacements |
| `{movie}_diffusion_coefficients.csv` | One row per particle (see below) |
| `{movie}_IMSD.csv`, `{movie}_EMSD.csv` | Per-particle and ensemble MSD vs lag |
| `{movie}_MSD_vs_tau.csv` | Combined MSD summary |
| `{movie}_track_metrics.csv` | One row per track, everything filterable |
| `{movie}_context.json` | Exact parameters used, lags computed, QC numbers |
| `qc_plots/` | Localization QC (below) |
| `{movie}_NO_TRACKS.json` | Only if the movie yielded nothing |

The diffusion table has `particle`, `n_spots`, `alpha` (the anomalous exponent
from the log-log MSD slope), `mass_per_frame`, and **`D_fit_um2_s` with its
`tau_fit_s`**. That D is `MSD(τ) / 4τ` at a *single* lag — the one you set with
`--diffusion-tau`, in frames. At 100 Hz, the default of 5 frames means 50 ms. D
without its τ isn't a number you can compare, which is why the lag travels with
it in every table.

### Localization QC

Written per movie to `qc_plots/` unless you pass `--no-qc-plots`. These are the
plots that let you justify a parameter choice rather than just record it.

| Figure | What it tells you |
|---|---|
| `..._qc_feature_properties.png` | Mass (log y), size, signal and eccentricity side by side, with your `--minmass` cutoff drawn on |
| `..._qc_subpixel_bias.png` | The fractional part of the fitted x and y positions. Flat is right. Peaks at 0 or 0.5 mean the centroid is snapping to pixel centres, which inflates short steps and pushes D up — the fix is a bigger `--diam` |
| `..._qc_ecc_vs_mass.png`, `..._qc_mass_vs_size.png` | Real spots form a tight cloud; the tails are debris, aggregates and merged spots |
| `..._qc_features_per_frame.png` | Detections over the movie, with a last/first decile ratio. A steady decline is bleaching; a sudden step is focus loss |
| `..._qc_track_lengths.png` | Track lengths with `--min-track-length` marked, so you can see what the filter threw away |
| `..._qc_detections_frame0100.png` | The detections drawn on the frame itself |

Pick the overlay frame with `--qc-detections-overlay-frames` (default 100; pass
several if you like). A frame past the end of a movie is skipped rather than
quietly swapped for a different one. The numbers behind these plots also go into
the context JSON, so you can compare movies without opening a single figure.

**Options:** `--mpp` (0.183), `--fps` (50), `--diam` (7), `--minmass` (40),
`--range` (5, linking search radius in pixels), `--memory` (0, frames a particle
may vanish for), `--min-separation` (3), `--percentile` (90),
`--min-track-length` (4), `--diffusion-tau` (5), `--channel` (0), `--tau`
(1 5 10), `--no-tau-steps`, `--no-qc-plots`,
`--qc-detections-overlay-frames` (100), `--n-jobs` (8), `--force`,
`--data-root`.

## Adding more lags: `compute tau-steps`

Step sizes at τ = 1, 5 and 10 frames come out of tracking automatically. This
command exists to add more lags later without re-tracking:

```bash
spt-pipeline compute tau-steps -m manifest.csv -o results/ --tau 20 40
```

A **step size** is how far a particle moved over a chosen interval τ. For each
τ, every detection at frame *f* is paired with the one at *f + τ* in the same
track, whenever both exist. A long track contributes many overlapping
displacements. Because diffusion in cytoplasm depends on the timescale you look
at, step sizes and D are always reported at a stated τ.

Existing files are skipped, so adding a lag only computes that lag.

**Options:** `--tau`, `--mpp`, `--fps`, `--min-nspots`, `--min-alpha`,
`--non-overlapping` (τ-spaced start frames — statistically independent, fewer of
them), `--n-jobs`, `--force`, `--data-root`.

## Expression markers: `analyze marker-intensity`

If you image an expression reporter alongside your tracking channel, this
measures how much reporter signal surrounds each track.

```bash
spt-pipeline analyze marker-intensity -m manifest.csv
```

Put `marker` in the `notes` column of the reporter image's manifest row. That's
all the setup needed — the stage also runs automatically at the end of
`track trackpy` when it sees marker rows.

### Finding the right marker for each movie

Tried in order, stopping at the first that gives exactly one answer:

1. **The `marker_image` column**, if you filled it in. Always wins.
2. **Same name plus a tag**: `488__001` ↔ `488__001_FICD`.
3. **Same trailing number, same folder**: `488__001.nd2` ↔ `560__001.nd2`. The
   prefix is the channel; the number is the field of view.
4. **A marker subfolder**: `C11-157_DMSO/488__001.nd2` ↔
   `C11-157_DMSO/FICD_marker/561-Exp__001.nd2`.
5. **Only one of each** in a folder.

Pairing never leaves the movie's own folder or its marker subfolder. Two
condition folders that both contain `488__001` are different cells that happen
to share a filename, and pairing across them would put one cell's expression
level on another cell's tracks. Anything ambiguous is reported and skipped, and
whatever it does work out gets written back into `marker_image` so you can check
or correct it.

### How the intensity is measured

The **mean over the detection disk**, not the one pixel under the localization.
A single pixel is dominated by shot noise, and rounding the position to the
nearest pixel ties the measurement to exactly the subpixel bias the QC plots
exist to catch. Averaging over the ~37 pixels of a `diameter=7` disk cuts noise
about six-fold and doesn't care about the rounding.

It's fast because the marker image is convolved *once* with a disk kernel,
giving an image where every pixel already holds its own disk mean. Assigning
values to a million localizations is then a single array lookup — milliseconds,
and exact, with no spatial binning. Disks hanging over the frame edge are
handled properly and reported as `frac_disk_in_frame`.

Each track gets one value, the mean along its path, plus the spread, its mean
and range of position, its distance to the nearest edge, and how many
localizations went into it. That's enough to filter later — by position, by
intensity, by edge overlap — without measuring anything again.

**Options:** `--diameter` (defaults to the `--diam` used at tracking),
`--marker-channel` (0), `--no-append` (write the per-track CSV but leave the
tracks and diffusion CSVs alone), `--force`, `--out-dir`, `--data-root`.

## Comparing conditions: `analyze compare-conditions`

```bash
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ [options]
```

Pools everything by condition and makes the comparison figures and statistics.
It reads only the small cached CSVs, never the big tracks files, so it runs in
seconds and you can re-run it as often as you like with different bins, filters
or groupings.

### Running one part at a time

```bash
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ --run-only marker
```

`--run-only` takes any of `steps`, `diffusion`, `msd`, `marker`, `advanced`
(aliases like `expression`, `ficd` and `step-size` work too). Each part reads
cached output, so redoing the expression analysis doesn't mean redoing step
sizes. A partial run writes `statistical_tests_partial.csv` rather than
overwriting the complete table with a subset of itself. A name it doesn't
recognise is an error, not a silent omission.

### What it produces

**Step sizes**, for each τ: four ways of summarizing the same displacements (see
[below](#how-step-sizes-are-summarized)), a cross-τ overlay per condition, and a
box plot of median step by condition carrying the significance marks. Every
figure states τ in frames and seconds and shares identical axes across
conditions, scaled to the full data so no tail is clipped.

**Diffusion**: D histograms on linear and log₁₀ axes, and
`D_boxplot_by_condition.png` — box is the IQR, whiskers the 10th–90th percentile
of the per-movie median D, each movie a dot coloured by imaging day, each day's
median a diamond. The numbers behind it are exported as `D_boxplot_*_medians.csv`
in both tidy and GraphPad-ready wide form.

**MSD**: ensemble MSD vs τ per condition with SEM across movies, on linear and
log-log axes, plus a box plot of MSD at the fit lag. The curve uses a
length-consistent set of tracks, so the same tracks contribute at every lag and
you don't get the artificial kink that appears at τ = `min_track_length` when you
average everything regardless of length.

**Expression markers**, when marker data exists — see
[below](#expression-marker-outputs).

### Log-scaled histogram axes

```bash
--hist-yscale both
```

Step-size distributions are heavy-tailed and the mobile population lives in the
tail, at fractions around 10⁻³ that are indistinguishable from zero on a linear
axis. On a log axis that tail is legible, and a two-population mixture shows up
as a change of slope rather than an invisible shoulder. `both` writes a
`*_logy.png` next to each linear figure.

Empty bins are left as gaps rather than floored to some small number, because a
floor invents a value the data doesn't contain. Worth knowing: on a log axis a
bin holding one event looks as prominent as one holding a thousand, so read the
tail's *slope*, not its individual bins.

### Expression-marker outputs

Under `marker_analysis/`, one folder per mobility measure (`D`, `step_tau1`,
`step_tau5`, `path_speed`), each containing:

- **Per-track scatter** of mobility against marker intensity, with a per-condition
  trend line and Spearman ρ in the legend.
- **Per-cell scatter** — each cell's median mobility against its median
  intensity. This is usually the biological question, because expression level
  is a property of a cell; within one cell, marker intensity mostly reports
  local ER density instead.
- **Expression bins**: mobility compared between conditions within each
  intensity quantile, which separates "this treatment changes mobility" from
  "cells in this treatment happen to express less".
- **Brightness bins**: the same comparison at matched track brightness, using
  absolute edges you set with `--brightness-bin-edges` so the boundaries are
  identical in every condition.

Plus `brightness/`, which distinguishes two things easy to confuse: **track
brightness** (`mass_per_frame`, how bright the tracked particle is — a detection
quality proxy and a known confounder of D) and **marker disk intensity** (the
reporter signal). Each gets a per-movie overlay within each condition, a pooled
per-condition histogram, a box plot with significance marks, and summary CSVs.
The per-movie overlay is the one to look at first: a pooled histogram can't tell
a genuinely bimodal condition from one where a couple of cells drag the median.

Reporting D *and* median step side by side is deliberate. On subdiffusive data,
D at a single lag mixes the size of the motion with how anomalous it is; the step
median doesn't. If the two panels agree, that's evidence. If they disagree, the
problem is in the metric, not the biology.

## Filtering tracks

Every filter takes a MIN and a MAX. Use `-` for an open end.

```bash
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ \
    --filter-brightness 400 - --filter-nspots 20 - --run-label bright_only
```

| Filter | On |
|---|---|
| `--filter-d` | diffusion coefficient, µm²/s |
| `--filter-velocity` | mean path speed, µm/s |
| `--filter-brightness` | track brightness (`mass_per_frame`) |
| `--filter-intensity` | marker intensity |
| `--filter-alpha` | anomalous exponent |
| `--filter-nspots` | localizations per track |
| `--filter-step` | median step at τ = 1 |
| `--filter-edge` | distance to the frame edge, px |
| `--filter-coverage` | fraction of the marker disk inside the frame |

Filters apply to one shared set of tracks, so no two figures in a results folder
can describe different data. `filter_report.csv` records how many tracks each
criterion removed, in order, and `filter_attrition_by_condition.csv` shows
survival per condition.

**That second file matters.** A filter keeping 90% of one condition and 10% of
another has quietly turned a comparison of treatments into a comparison of
selection effects. The run prints a warning when attrition differs by more than
25 percentage points across conditions.

A missing value means "not measured", so those tracks are kept by default; pass
`--filter-drop-missing` to drop them instead. A filter naming a column your data
doesn't have is reported as not applied, rather than silently doing nothing.

`--run-label NAME` writes the pass to `compare_conditions/NAME/`, so you can
filter, look at the result, filter differently, and still have the first one to
compare against.

### Where the filterable numbers come from

`{movie}_track_metrics.csv`, one row per track, built from files that already
exist so it costs seconds:

| Column | What it is |
|---|---|
| `D_fit_um2_s`, `tau_fit_s` | D at a single lag, and that lag |
| `v_path_um_s` | Mean speed along the path: total distance over time observed |
| `step_median_um_tau{N}` | Median displacement at lag N |
| `alpha`, `n_spots` | Anomalous exponent, localizations per track |
| `mass_per_frame` | Track brightness |
| `marker_intensity` + position and quality | From the marker stage |

Three mobility measures rather than one, because they disagree in useful ways.
Mean path speed is intuitive but the most inflated by localization error — every
noise wobble adds path length, and a particle sitting still still accrues speed.
At 100 Hz with 40 nm precision that's roughly 8 µm/s of pure noise. The step
medians are robust and need no fit, and comparing a short lag against a long one
tells you how much of your effect is noise: if it holds at τ = 1 and dies at
τ = 5, it was noise.

## How step sizes are summarized

Four views of the same displacements, because each answers a different question
and they can disagree:

- **Raw pooled** — every displacement counted once. The primary view. Long
  tracks contribute more, which is right if you care about the population of
  displacements.
- **Track-weighted** — each track's own histogram, then averaged with SEM
  ribbons. Every track gets one vote regardless of length. Use it to check that
  the pooled view isn't being driven by a handful of long tracks.
- **Track-mean** — one value per track. The coarsest, and the most robust to
  track-length differences.
- **Track-length split** — pooled, but stratified by track length, so you can
  see whether short and long tracks behave differently. They usually do, and
  that's a survivorship effect worth knowing about.

Bin widths are chosen automatically and floored at the localization precision
(`mpp/10`), so you're never plotting structure finer than you can measure.
Override with `--step-bin-width-um`, `--diffusion-bin-width`, `--logd-bin-width`.

## How conditions are compared

A track is the thing you measure. An **experiment** — one imaging day, one
biological replicate — is the thing you do statistics on. A condition is what
you compare.

### The design is chosen from your manifest

That hierarchy only works if you actually have replicate experiments. Run a
single imaging day through an experiment-level test and every condition has
n = 1, giving p-values that are undefined or meaningless — and printed on a
figure, worse than nothing.

So the pipeline reads the `experiment` column and decides:

| Your data | Design | Unit | A significant result means |
|---|---|---|---|
| Every condition has ≥ 2 experiments | between-experiment | experiment | The effect reproduces across imaging days |
| Any condition has 1 | within-experiment | movie | The conditions differ *within this experiment*; reproducibility isn't addressed |

The gate is the *minimum* number of experiments across conditions, not the
total: two experiments split 2-and-1 still leaves one arm at n = 1, so that also
falls back. The decision and its reason go into
`compare_conditions/statistical_design.json`, into every row of
`statistical_tests.csv`, and onto the figures.

### The tests

- **Mann-Whitney U** on per-unit summaries between conditions. Nonparametric,
  which suits small n.
- **Cliff's delta** for effect size, and **Benjamini-Hochberg** correction when
  there are more than two conditions.
- **Bootstrap confidence intervals** resampled at the statistical unit.
- **Track-level tests** (KS, Mann-Whitney, Cliff's delta on pooled per-track
  values) always computed, written to `track_level_ks_tests.csv`, and
  deliberately never put on a figure. With n in the hundreds of thousands
  everything is significant; read `cliffs_delta`, which doesn't inflate with n.

### Localization-error subtraction

```bash
--subtract-loc-error
```

Estimates the noise floor from quasi-immobile particles — low alpha, enough
localizations — whose short-lag MSD is essentially 4σ²loc, and subtracts it. Done
per movie, because the floor is a property of that acquisition.

It reports what it did: the per-movie floor and implied σ_loc go to
`loc_error_floor_by_movie.csv`, and if the diffusion tables lack the columns it
needs, it says so instead of quietly doing nothing. It also warns if the
correction drives more than 40% of tracks to D = 0, which means the "immobile"
population wasn't actually immobile and the floor came out too high. Tune with
`--loc-error-alpha-max` (0.3) and `--loc-error-min-nspots` (10).

### Figures

One colourblind-safe palette (Okabe-Ito), assigned from a sorted list so a
condition keeps its colour on every panel. Significance is drawn as comparison
bars with asterisks or `n.s.` — no test names, no numeric p-values, no
explanatory text. The exact statistics are in the CSVs, where you can read them
properly. Bars are stacked so they never overlap, and thinned automatically when
there are too many pairs to show clearly. 300 dpi.

## Advanced trajectory analyses

```bash
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ --advanced
spt-pipeline analyze compare-conditions -m manifest.csv -o results/ --advanced-only VACF a2
```

Six analyses beyond step sizes and D, run per condition and then compared:

| Keyword | What it measures | Also called |
|---|---|---|
| `vacf` | Velocity autocorrelation, multi-δ | `velocity` |
| `cfbm` | Fractional-Brownian fit to the collapsed VACF (Hurst exponent) | `fbm`, `hurst` |
| `angles` | Turning angles, ⟨cos θ⟩ vs τ | `cos_theta`, `turning` |
| `eb` | Ergodicity-breaking parameter | `ergodicity` |
| `ngp` | Non-Gaussian parameter α₂ | `a2`, `alpha2` |
| `rheology` | Complex shear modulus G′/G″, Mason/GSER | `microrheology`, `gser` |

Keywords are case-insensitive and an unrecognised one is an error.
`--advanced-only` implies `--advanced`. Conditions come from the manifest, and
coordinates are converted to microns using each movie's own `mpp`, so a dataset
acquired at two pixel sizes is handled correctly.

Track-length and lag settings default to whatever tracking used, rather than a
separate hardcoded value, so the same tracks are analysed throughout. Override
with `--advanced-min-track-len` and `--advanced-max-tlag`. Rheology needs a probe
radius: set `--advanced-particle-radius` in microns, or you'll get a one-time
reminder of the 20 nm default, since G′/G″ scale with it.

## Output structure

```
results/
├── {movie}/                          # one folder per movie
│   ├── {movie}_tracks.csv
│   ├── {movie}_tau_steps_001.csv     # and 005, 010, …
│   ├── {movie}_diffusion_coefficients.csv
│   ├── {movie}_track_metrics.csv     # one row per track, everything filterable
│   ├── {movie}_marker_intensity.csv  # if you have marker images
│   ├── {movie}_IMSD.csv, _EMSD.csv, _MSD_vs_tau.csv
│   ├── {movie}_context.json          # exact parameters + QC numbers
│   ├── {movie}_NO_TRACKS.json        # only if nothing was found
│   └── qc_plots/
├── run_logs/                         # one JSON per command you ran
└── compare_conditions/[<run-label>/]
    ├── run_parameters.json           # including the exact filters used
    ├── statistical_design.json       # which unit, and why
    ├── statistical_tests.csv         # unit-level tests, the ones on the figures
    ├── track_level_ks_tests.csv      # pooled-track tests, never on a figure
    ├── filter_report.csv             # attrition per criterion
    ├── filter_attrition_by_condition.csv
    ├── loc_error_floor_by_movie.csv  # with --subtract-loc-error
    ├── condition_summary.csv
    ├── step_size_analysis/tau_001/ … /cross_tau/
    ├── diffusion_analysis/
    ├── msd_analysis/
    ├── marker_analysis/{D,step_tau1,step_tau5,path_speed}/ and brightness/
    └── advanced_analysis/            # with --advanced
```

## Re-running efficiently

The expensive step is tracking, and it's cached. Everything downstream reads
small CSVs.

- **Change a bin width, a filter, or a grouping** → re-run
  `analyze compare-conditions`. Seconds.
- **Add a lag** → `compute tau-steps --tau 20`. Only the new lag is computed.
- **Redo just the expression analysis** → `--run-only marker`.
- **Try a different filter** → add `--run-label` and keep both results.
- **Re-track after changing detection settings** → `track trackpy --force`.
- **Interrupted run** → just run it again; finished movies are skipped.

## Reproducibility

Every command writes a JSON run log with all its parameters, the manifest it
used, and how many movies it processed and skipped. Every per-movie folder has a
`context.json` with the exact tracking parameters and QC numbers. Every
comparison folder records its bin settings, its statistical design, and — if you
filtered — the exact windows and their attrition. A figure with a filtered
population whose selection you can't reconstruct isn't a result, so the pipeline
makes sure you can always reconstruct it.

## Companion tools

**`spt-coloc`** — two-colour co-diffusion. Registers channels from TetraSpeck
beads, pairs tracks across channels, and tests co-movement against
frame-shuffled controls.

```bash
spt-coloc register beads.tif -o registration/
spt-coloc pair ch1/ ch2/ registration/transform.json --mpp 0.122 --fps 50 -o coloc/
spt-coloc analyze coloc/ --n-shuffles 500 -o coloc/
```

**`spt-metadata`** — walks a directory of ND2/TIFF movies and writes one CSV row
per movie with the acquisition settings: frames, exposure, pixel size,
objective, lasers. Useful for checking that two imaging sessions really were
comparable before you compare them.

```bash
spt-metadata <dir> [--output metadata.csv]
```

**Still available** under the older interface: `prep trackmate`,
`prep nd2-to-tiff`, `prep stack-channels`, `prep find-ome-tif`,
`draw movie-tracks`, `utils split-metrics`.

**Removed in 2.1**: the brightness-mobility, step-brightness and dwell-time
analyses, and the whole `plot` command group. The code is preserved under
`retired/` in the repository.

## Parameter reference

### Detection and linking

| Parameter | Units | Typical | What it does |
|---|---|---|---|
| `--diam` | pixels | 5–15, odd | Expected spot size. Too small causes subpixel bias — check the QC plot |
| `--minmass` | a.u. | 20–200 | Brightness cutoff. Check it against the mass histogram |
| `--percentile` | % | 80–95 | Only consider local maxima above this percentile |
| `--min-separation` | pixels | 3–7 | Minimum distance between two detections |
| `--range` | pixels | 3–15 | How far a particle may move between frames |
| `--memory` | frames | 0–3 | How long a particle may vanish and still be the same track |
| `--min-track-length` | frames | 4–20 | Shorter tracks are discarded |

### Measurement and comparison

| Parameter | Units | Typical | What it does |
|---|---|---|---|
| `--tau` | frames | 1–20 | Lags at which displacements are measured (default 1 5 10) |
| `--diffusion-tau` | frames | 3–10 | The lag D is computed at |
| `--mpp`, `--fps` | µm/px, Hz | — | Calibration; per-movie values in the manifest win |
| `--subtract-loc-error` | flag | — | Subtract the localization noise floor from D |
| `--step-bin-width-um` | µm | auto | Histogram bin width, floored at `mpp/10` |
| `--hist-yscale` | — | linear | `linear`, `log`, or `both` |
| `--gmm` | flag | — | Fit D subpopulations, component count by BIC |
| `--n-jobs` | count | 1–N | Parallel worker processes |

## Troubleshooting

**`FILL_IN` validation error** — open the manifest and set the flagged
`condition` / `experiment` cells.

**`trackpy` or `nd2` required** — `pip install "spt-pipeline[tracking,imaging]"`.

**No particles detected** — lower `--minmass` or `--percentile` and check
`--diam` against your spot size. Look at `qc_plots/..._qc_mass_histogram` to see
where the cutoff actually falls.

**A movie reports "no tracks"** — not an error. Read its `NO_TRACKS.json` for
the reason and the thresholds, then re-run that movie with `--force` and looser
settings.

**D looks too high, or short steps look inflated** — check
`qc_plots/..._qc_subpixel_bias.png`. Peaks at 0 or 0.5 mean pixel biting; raise
`--diam`. Then consider `--subtract-loc-error`.

**Source paths not found** — if the manifest came from another machine, pass
`--data-root` pointing at where the data is mounted now. The error message lists
what it tried.

**Statistics say "technical replicates"** — you have one experiment for at least
one condition, so the movie is the statistical unit. Put real imaging dates in
the `experiment` column to get biological-replicate inference.

**A filtered comparison looks too good** — check
`filter_attrition_by_condition.csv`. If the filter kept very different fractions
of each condition, you're comparing selection effects.

**Nothing to compare, or empty figures** — check that `*_tau_steps_*.csv` files
exist beside each movie's tracks and that the manifest's `tracks_csv` paths
resolve.

## License and citation

MIT — see [LICENSE](LICENSE). If this supports a publication, please cite it
along with [trackpy](http://soft-matter.github.io/trackpy/). Step-size and
diffusion-at-τ conventions follow established Holt-lab SPT analyses.
