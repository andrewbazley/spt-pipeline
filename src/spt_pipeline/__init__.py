"""SPT pipeline: manifest-driven single-particle-tracking analysis.

Four cacheable stages, heaviest first, each reading what the previous one
wrote:

* **prep scan** (:mod:`~spt_pipeline.manifest`) -- walk your data and build the
  manifest CSV that every later stage reads.
* **track trackpy** (:mod:`~spt_pipeline.tracking`) -- detect and link
  particles per movie, then compute fixed-tau displacements
  (:mod:`~spt_pipeline.tau_steps`), the diffusion table
  (:mod:`~spt_pipeline.diffusion`) and localization QC
  (:mod:`~spt_pipeline.qc`). Run once; results are cached.
* **analyze marker-intensity** (:mod:`~spt_pipeline.marker`) -- measure
  expression-reporter signal around every track, from paired marker images.
* **analyze compare-conditions** (:mod:`~spt_pipeline.compare`) -- pool by
  condition and produce the figures and statistics. Reads only cached CSVs, so
  it re-runs in seconds under different filters.

The manifest is the single source of truth for what every movie is. Its paths
are stored relative to itself, so a dataset moves between Windows, macOS and
Linux without editing.

Supporting modules: :mod:`track_metrics` (one row per track, everything
filterable), :mod:`filters` (declarative track selection),
:mod:`statistics` (design inference and tests), :mod:`plotting` (one palette,
one house style), :mod:`histograms`, :mod:`bins`, :mod:`gmm`,
:mod:`advanced` (VACF / CFBM / angles / EB / alpha_2 / rheology),
:mod:`run_log`, :mod:`utils`.

Console scripts
---------------
``spt-pipeline`` -> :func:`spt_pipeline.cli.main`
``spt-coloc``    -> :func:`spt_pipeline.colocalization.main`
``spt-metadata`` -> :func:`spt_pipeline.metadata.main`
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("spt-pipeline")
except PackageNotFoundError:  # editable install / not yet installed
    __version__ = "2.1.0"

__all__ = ["__version__"]
