# Validation summary

## Supplied failed test run

- 32 marker images were listed in the four `run_metadata*.csv` files.
- 0/32 had a nonempty `trajectory_file`.
- All 32 used `segmentation_source=red_only`.
- 11/32 produced no cell row.
- The remaining images produced 29 candidate cell rows, but only 1/29 had
  `qc_pass=True`.
- 10/29 candidate regions were smaller than 30 um2 and 20/29 touched an image
  boundary.

The direct cause was pairing-key loss: the old script called `resolve()` on
staged `Red_<movie_id>.nd2` symlinks, changing the red key back to the original
marker basename. The trajectory map retained `Traj_<movie_id>`, so the keys no
longer matched and the script silently used red-only segmentation.

## Real paired regression

Input:

- `Red_A8_EG_0.1pcDMSO_009.nd2`
- `Traj_A8_EG_0.1pcDMSO_009.csv`

Results with v2 defaults:

- Background peak: 103.0
- Background lower-side sigma: 2.965
- Trajectories with >5 steps used as seeds: 678
- Qualifying seed tracks cleanly assigned: 674/678
- Inferred cells: 1
- Track-seed area: 577.35 um2
- Final measurement area: 898.90 um2
- Growth expansion ratio: 1.56
- Puncta-excluded background-subtracted mean: 21.37
- Marker status: `detected_with_perimeter`
- Perimeter z-score: 1.13
- Puncta fraction: 1.40%
- Technical QC: pass

## Synthetic no-signal regression

The same real track cloud was placed over a simulated marker image containing
camera-like background centered at 103 plus 80 compact bright puncta.

- Inferred cells: 1
- Marker status: `no_signal`
- Background-subtracted mean: 0.014
- Puncta fraction: 2.33%
- Technical QC: pass

This verifies that a marker-negative image can return a valid near-zero value
without requiring a red-channel perimeter.

## Remaining validation need

Only one real paired raw marker/trajectory example was available in this
workspace. The supplied 32-image test results did not include their raw ND2
images, trajectory CSVs, masks, or QC overlays. A representative rerun should
therefore be reviewed before freezing parameters, especially for multiple
cells, adjacent cells, very sparse long tracks, apoptotic cells, and fields in
which marker signal occupies most of the image.
