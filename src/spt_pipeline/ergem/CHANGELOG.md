# pipeline patch (post-v3.1b)

- Fixed a crash in `d_boxplot` (figures): per-movie points, replicate colours,
  and box stats are now taken from the same NaN-filtered rows, so a movie whose
  D could not be computed no longer raises a colour/size mismatch.

# pipeline patch (post-v3.1)

- Expression QC is now per-image: an unpaired/failed field no longer discards
  its whole experiment; only a fully-unpaired experiment is an error. The run
  log records `n_paired_images` / `n_unpaired_images` per experiment.
- `ergem run` always runs cross-correlation (folder is populated even when
  empty) and always attempts the D-vs-FICD figures, adding a per-cell figure
  (`Dcell_vs_ficd_expression.png`); the run prints expression/correlation counts.

# v3.1.0 (adaptive-track-territory)

- Canonical expression column renamed `ficd_bgsub_mean` -> `ficd_bgsub_intensity`
  (the old name is retained as an equal alias for backward compatibility; the
  pipeline's `FICD_VALUE_COL` and figures follow the new name automatically).
- Replaced intensity-threshold flood growth with adaptive 3-px dilation-ring
  growth: a ring is rejected on a >=1 robust-SD median shift or >40% incompatible
  pixels, capped at 3x the seed area.
- Requires >=100 qualifying seed trajectories (>5 steps) per territory; adds
  track clustering (radius/min-neighbors) and compact-punctum removal before
  growth and measurement.
- Marker-negative territories report an intensity of exactly 0.
- Added `--version` (`3.1.0-adaptive-track-territory`); legacy v2 params kept as
  hidden `argparse.SUPPRESS` fallbacks and legacy functions preserved as
  `_legacy_*`.

# v2 changes

- Fixed staged marker/trajectory pairing by preserving `Red_<movie_id>` symlink
  basenames instead of resolving marker links to their original filenames.
- Validates paired input counts after every experiment-level run.
- Requires a trajectory file for every marker image; red-only fallback is gone.
- Uses tracks with more than five steps as spatial seeds.
- Replaced local-annulus background subtraction with a full-image histogram
  background peak and lower-side spread estimate.
- Added track-seeded, intensity-constrained outward growth.
- Added a valid `no_signal` phenotype with a track-defined measurement region.
- Added compact bright-punctum exclusion.
- Separated technical `qc_flags` from biological/geometry `phenotype_flags`.
- Added seed masks and four-panel QC plots including the pixel histogram.
- Added cell-level diffusion/expression tables and correlation summaries while
  preserving the original per-track outputs.
