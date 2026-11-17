"""FICD complementation ER-GEM analysis (subpackage).

This subpackage layers a FICD-specific workflow on top of the core
``spt_pipeline`` without changing any existing behaviour:

* :mod:`.condition_map` -- read a ``*_condition_map.json`` (or the older
  ``*_dataset_map`` layout) emitted by the LUMEN organisational pipeline and
  flatten it into the core manifest schema (plus additive factor columns).
* :mod:`.ingest` -- turn each movie's MOSAIC ``Traj_*.csv`` into the same
  per-movie diffusion / MSD products the trackpy path produces, so the standard
  ``compute tau-steps`` and ``analyze compare-conditions`` phases run unchanged.
* :mod:`.expression` -- drive the per-cell FICD marker measurement
  (``ficd_cell_intensity``) from the explicit red/trajectory pairs in the map.
* :mod:`.crosscorr` -- join per-track diffusion to per-track FICD expression and
  emit the diffusion-vs-expression cross-correlation table.
* :mod:`.contrasts` -- the factored WT-vs-E234G x DMSO-vs-Tg comparison, pooled
  and clone-resolved, with an optional mixed-effects model.

Nothing here is imported by the core package; the core ``spt-pipeline`` CLI is
untouched. Use ``python -m spt_pipeline.ergem`` for the FICD commands.
"""
from __future__ import annotations

__ergem_version__ = "3.1.0-adaptive-track-territory"  # track-seeded FICD expression (see CHANGELOG.md)

DEFAULT_MPP = 0.11    # micrometres per pixel for the ER-GEM campaign
DEFAULT_FPS = 100.0   # frames per second for the ER-GEM campaign

# Additive manifest columns carried alongside the core schema.
EXTRA_COLUMNS = [
    "genotype", "complementation", "treatment_code", "clone",
    "has_paired_red", "ficd_expr_table", "ficd_marker_nd2", "condition_id", "replicate_id",
]

__all__ = ["DEFAULT_MPP", "DEFAULT_FPS", "EXTRA_COLUMNS"]
