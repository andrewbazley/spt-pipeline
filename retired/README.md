# Retired code

Code removed from `spt-pipeline` but kept for reference. **Nothing here is
imported by the package, and nothing here is expected to run as-is.** These
files were moved, not rewritten: no attempt has been made to make them
standalone, resolve their imports, or keep them working.

## The brightness / dwell analysis arm

Retired because it represents an analysis direction no longer in use. It used
trackpy, but was not really a tracking analysis.

| File | Contents |
|------|----------|
| `brightness_dwell_arm.py` | The seven command functions, verbatim from `spt_pipeline/legacy.py` |
| `cli_fragments.py` | The argparse blocks that registered them, verbatim from `legacy.main()` |
| `brightness.py` | The v2 `spt_pipeline.brightness` module (flatfield correction, brightness–diffusion correlation) |
| `brightness_plot.py` | `brightness_d_scatter`, from `spt_pipeline.plotting` |

Commands that no longer exist:

- `analyze brightness-mobility`
- `plot steps-brightness`
- `plot dwell`
- `plot brightness-heatmap`
- `plot brightness-map`
- `track dwelltime`
- `workflow legacy-auto` (the chain of the first three)

The whole `plot` command group is gone, since all four of its subcommands were
in this arm.

### Also removed, as dependents of `brightness.py`

- `track trackpy --correct-brightness` — the 2D-polynomial flatfield correction
  of per-track brightness.
- `analyze compare-conditions --brightness` — the brightness–diffusion
  correlation and the `brightness_analysis/` output directory.

These are worth a second look. The brightness–diffusivity relationship (dimmest
particles most immobile, surviving per-movie z-score normalization) is a live
question rather than a dead analysis arm, and it was retired here only because
it shares `brightness.py` with the retired code. Reviving it means lifting
`flatfield_correct`, `add_brightness_zscore`, and
`brightness_diffusion_correlation` back into the package — those three functions
have no dependency on anything else in this folder.

## Still in the package but unreachable

Not moved, because removing them was not asked for, but flagged as the next
obvious cleanup: `spt_pipeline/legacy.py` still contains
`cmd_track_trackpy` (~775 lines) and `cmd_analyze_compare_conditions`
(~691 lines). Both are shadowed by their v2 replacements in `cli.py`, which
intercepts `track trackpy` and `analyze compare-conditions` before the legacy
dispatcher ever sees them, so neither can currently be invoked.
