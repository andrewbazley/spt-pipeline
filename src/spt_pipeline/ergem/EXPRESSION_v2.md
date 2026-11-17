# Track-seeded FICD expression measurement (v2)

This is a drop-in update for the `spt_pipeline.ergem` expression workflow. It
keeps the existing output filenames and the primary `ficd_bgsub_mean` column,
while changing the measurement geometry from red-only segmentation to
track-seeded outward growth.

Replace these files in `spt_pipeline/ergem/`:

- `_ficd_cell_intensity.py`
- `expression.py`
- `crosscorr.py` (additive cell-level correlation outputs)

No `cli.py` change is required.

## Measurement logic

1. Retain trajectories with strictly more than five steps (at least seven
   localizations) as cell-seeding evidence.
2. Rasterize all localizations from those trajectories with equal total weight
   per trajectory, smooth the density, dilate it, and bridge nearby ER regions.
3. Estimate the dominant background peak and its lower-side spread from the
   full marker-image histogram. The default peak hint is 105 intensity units,
   with a +/-30-unit search window; the peak is not hard-coded to 105.
4. Probe the marker image around each track seed. If diffuse signal is detected,
   grow through connected pixels above `background + 1.5 sigma` until the
   frontier becomes background-like.
5. If diffuse signal is not detected, retain a conservative dilated track
   territory and report `marker_status=no_signal`. This is a valid phenotype,
   not automatically a QC failure.
6. Detect and exclude compact high-frequency bright puncta.
7. Measure the arithmetic mean in the resulting region and subtract the global
   histogram background peak.
8. Assign that one cellular value to all trajectories spatially mapped to the
   region.

## Important outputs

- `cells.csv`: one row per inferred cell/track territory.
- `trajectory_expression.csv`: one row per trajectory with the cellular FICD
  value appended.
- `localizations_with_expression.csv`: localization-level assignment table.
- `run_metadata.csv`: pairing, background-peak, seed, and acquisition metadata.
- `masks/*_track_seeds.tif`: the ER-GEM-derived seed labels.
- `masks/*_labels.tif`: final measurement regions.
- `qc/*_qc.png`: raw marker, seeds/density, final mask/puncta, and histogram.

`marker_status` is one of:

- `detected_with_perimeter`
- `detected_no_complete_perimeter`
- `no_signal`
- `manual_override`

`qc_pass` now describes technical validity. A row can therefore have
`marker_status=no_signal`, `phenotype_flags=no_marker_signal`, and
`qc_pass=True`.

## Pipeline execution

The existing command remains valid:

```bash
python -m spt_pipeline.ergem expression \
  -m /path/to/manifest.csv \
  --condition-map /path/to/condition_map.json \
  -o /path/to/results \
  --n-jobs 8
```

Optional measurement parameters can still be passed after `--`, for example:

```bash
python -m spt_pipeline.ergem expression \
  -m /path/to/manifest.csv \
  --condition-map /path/to/condition_map.json \
  -o /path/to/results \
  -- --growth-z 1.75 --track-bridge-radius-um 3.0
```

The defaults should be treated as a starting point until QC overlays from a
representative set of bright, dim, marker-negative, multi-cell, and apoptotic
fields have been reviewed.

## Cross-correlation

The original per-track outputs are preserved. The revised `crosscorr.py` also
writes:

- `Dcell_vs_ficd_expression.csv`
- `correlation_by_condition_cells.csv`
- `correlation_by_experiment_cells.csv`

These aggregate diffusion within each `(movie_id, cell_id)` before correlation,
avoiding inferential pseudoreplication of one expression value across many
tracks. Per-track tables remain available for visualization and hierarchical
models.

## Dependencies

The dependencies are unchanged from v1:

```bash
python -m pip install nd2 numpy pandas scipy scikit-image tifffile matplotlib
```
