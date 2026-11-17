# Changelog

## 2.1.0

### Manifests move between computers

Paths in the manifest are now stored relative to the manifest file itself, in
forward-slash form. A manifest and its data can move between Windows, macOS and
Linux without editing.

Reading is more forgiving than writing. If a stored path doesn't resolve — an
old manifest full of `\\server\share\...`, a dataset now mounted somewhere else
— the loader matches the deepest tail of that path underneath the manifest's own
directory. `//research-cifs/…/20260805 Expt/C11-157_DMSO/488__001.nd2` is found
at `C11-157_DMSO/488__001.nd2` regardless of what sat above it before. Longest
match wins, and a bare filename is never matched by searching, so it can't grab
the wrong file. `--data-root` overrides the anchor when the data lives away from
the manifest.

### Movies that produce nothing are skipped, not fatal

A movie with no particles above threshold, or no track long enough to survive
the length filter, now returns `no_tracks` instead of `failed`. The run
continues, a `{movie}_NO_TRACKS.json` records the reason and the thresholds in
force, and every later stage skips that movie with a warning. A path *listed* in
the manifest but missing from disk is still a hard error.

Marker images and any single-frame acquisition are excluded from tracking
automatically — there's no time axis to link along.

### Tracking does more in one pass

`track trackpy` now also computes fixed-tau displacements (τ = 1, 5, 10 frames
by default) and writes localization QC figures. `compute tau-steps` remains, for
adding further lags later without re-tracking.

New QC panel set per movie, in `qc_plots/`: a combined mass/size/signal/ecc
strip with log-y mass, the x/y subpixel-bias strip, ecc-vs-mass, mass-vs-size,
features per frame, track lengths, and a detections overlay on frames chosen
with `--qc-detections-overlay-frames` (default 100). Scalar versions of the same
diagnostics go into the per-movie context JSON so movies can be compared without
opening figures.

### The statistical design is chosen from your data

`analyze compare-conditions` reads the manifest's `experiment` column and picks
the unit of inference itself:

- every condition has ≥ 2 experiments → **between-experiment**, unit = experiment
- any condition has 1 → **within-experiment**, unit = movie, labelled technical

The gate is the *minimum* experiments per condition, so a 2-versus-1 split still
falls back rather than testing an n = 1 arm. The decision and its reason land in
`statistical_design.json`, in every row of `statistical_tests.csv`, and on the
figures.

Track-level distributional tests (KS, Mann-Whitney, Cliff's delta on pooled
per-track values) are always computed and written to `track_level_ks_tests.csv`,
and never annotated on a figure — with n = tracks, everything is significant.

### Expression-marker (FICD) analysis

New. Pairs each movie with its expression-reporter image and measures the mean
marker signal over the detection disk at every localization.

Pairing order: the `marker_image` column if set, then stem appendage
(`488__001` ↔ `488__001_FICD`), then trailing index within the same directory
(`488__001` ↔ `560__001`), then a marker subfolder of the movie's folder
(`C11-157_DMSO/FICD_marker/561-Exp__001`), then sole occupant. Pairing never
leaves the movie's own folder or its marker subfolder — two condition folders
both containing `488__001` are different fields of view. Ambiguity is reported,
never guessed, and the resolved pairing is written back to `marker_image`.

Intensity is the **disk mean**, not the single pixel under the localization: one
convolution of the marker image with a normalized disk kernel gives a
full-resolution image where every pixel already holds its disk mean, so
assignment is one vectorized lookup. Exact, no spatial binning, milliseconds
regardless of track count. Partial disks at the frame edge are handled by
convolving a validity mask and reported as `frac_disk_in_frame`.

One value per track, plus position, spread and quality columns so tracks can be
filtered afterwards without re-measuring. Outputs cover the per-track scatter,
the per-cell scatter (expression is a property of a cell, so this is the
biological question), expression bins, brightness bins, and brightness
distributions per movie and per condition.

`analyze marker-intensity` runs this on its own, and it also runs automatically
at the end of `track trackpy` when marker rows exist.

### Filtering, and re-running under new filters

New per-track metrics table, `{movie}_track_metrics.csv`, assembled from files
that already exist: D and its `tau_fit_s`, mean path speed, median step at every
cached tau, alpha, n_spots, brightness, marker intensity.

`--filter-d`, `--filter-velocity`, `--filter-brightness`, `--filter-intensity`,
`--filter-alpha`, `--filter-nspots`, `--filter-step`, `--filter-edge`,
`--filter-coverage`, each taking MIN and MAX with `-` for an open bound. They
apply to one shared track population, so no two figures in a folder can describe
different tracks. `filter_report.csv` records attrition per criterion, and
`filter_attrition_by_condition.csv` exposes uneven selection — a filter keeping
90% of one condition and 10% of another has turned a treatment comparison into a
selection-effect comparison, and the run says so.

`--run-label NAME` writes each pass to its own subtree, so you can filter, look,
re-filter and still have the earlier result.

### Running one arm at a time

`--run-only` takes any of `steps`, `diffusion`, `msd`, `marker`, `advanced`,
with aliases. Each arm reads cached per-movie output, so re-doing the expression
analysis no longer means re-doing step sizes. A partial run writes
`statistical_tests_partial.csv` so it can't overwrite the complete table.

### Localization-error subtraction now reports itself

`--subtract-loc-error` previously degraded to a silent no-op when the diffusion
tables lacked `msd_fit_um2`. It now either applies and reports the per-movie
floor and implied σ_loc (`loc_error_floor_by_movie.csv`), or says exactly what's
missing. It also warns when the correction drives more than 40% of tracks to
D = 0, which means the "immobile" population used to estimate the floor wasn't
immobile.

### Figures

- Condition colours are assigned from a sorted list, so a condition keeps its
  colour on every panel.
- Significance is drawn as comparison bars with asterisks or `n.s.`, tiered so
  they never cross, thinned automatically past eight pairs. No test names, no
  numeric p-values, no interpretive text anywhere.
- `--hist-yscale {linear,log,both}` puts the fraction axis on a log scale, which
  is where the mobile tail becomes legible. Empty bins are masked, leaving gaps,
  rather than floored to an invented value.
- New advanced suite (`--advanced`, `--advanced-only`): velocity
  autocorrelation, fractional-Brownian collapse, turning angles,
  ergodicity-breaking, non-Gaussian parameter α₂, and Mason/GSER microrheology.
  Grouped by the manifest, not by filename.
- `analyze compare-conditions` gained a per-tau `median_step_boxplot.png` and an
  `msd_at_fit_boxplot.png` to carry significance that the distribution overlays
  can't.

### Renamed

- `D_violin_by_condition.png` → `D_boxplot_by_condition.png`; the median CSVs
  gained a matching `D_boxplot_` prefix. `plotting.d_violin` is gone — it never
  drew a violin.
- `workflow auto` now means the v2 core path: scan → track → compare.

### Removed

The brightness-mobility / step-brightness / dwell analysis arm, and the whole
`plot` command group with it. Preserved verbatim under `retired/` at the
repository root, along with `brightness.py`, the flatfield correction
(`track trackpy --correct-brightness`) and the brightness–diffusion correlation
(`compare-conditions --brightness`), which depended on it.

### Packaging

- Core dependencies trimmed to numpy, pandas, matplotlib and scipy, so cached
  output can be re-analysed without the tracking or imaging stacks.
  scikit-learn, scikit-image and statsmodels moved to the `gmm`, `coloc` and
  `stats` extras; all three are imported lazily.
- Fixed the `readme` filename in the project metadata, which pointed at a file
  that did not exist.

## 2.0.0

Consolidation of three divergent monolithic scripts into one modular package:
manifest-driven metadata, vectorized fixed-tau displacements, the four-histogram
representation hierarchy, experiment-level statistics, and the two-colour
co-diffusion and metadata companion tools.
