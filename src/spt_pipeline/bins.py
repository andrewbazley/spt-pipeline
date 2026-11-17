"""Physical-unit histogram bin infrastructure.

A single source of truth for how displacement, diffusion, and log-scale
histograms are binned, ported and unified from
``pipeline_fraction_bins_compare_tau.py``. Bin widths can be user-specified or
auto-derived, and are always floored to a physically meaningful minimum
(localization precision for step sizes) so that no figure implies a resolution
the microscope cannot support.

All bin settings are captured in a dict suitable for JSON export
(``histogram_bin_settings.json``) for full reproducibility.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def resolve_bin_width(
    requested: Any,
    default_value: float,
    min_reasonable: Optional[float] = None,
    label: str = "bin",
) -> Tuple[float, str]:
    """Resolve a requested/auto bin width, enforcing a physical floor.

    Returns ``(width, source)`` where ``source`` documents provenance for the
    settings JSON. A width below ``min_reasonable`` (e.g. mpp/10 for step
    sizes) is raised to that floor.
    """
    if requested is None or str(requested).strip().lower() == "auto":
        width = float(default_value)
        source = "auto default"
    else:
        width = float(requested)
        source = "user"
    if (
        min_reasonable is not None
        and np.isfinite(min_reasonable)
        and min_reasonable > 0
        and width < min_reasonable
    ):
        width = float(min_reasonable)
        source += " + parameter floor"
    return width, source


def make_bins(xmin: float, xmax: float, width: float) -> np.ndarray:
    """Evenly spaced bin edges in physical units, guaranteed to span [xmin,xmax]."""
    xmin, xmax, width = float(xmin), float(xmax), float(width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"Invalid bin width: {width}")
    if xmax <= xmin:
        xmax = xmin + width
    bins = np.arange(xmin, xmax + width * 0.5, width)
    if len(bins) < 2 or bins[-1] < xmax:
        bins = np.append(bins, (bins[-1] + width) if len(bins) else xmax)
    return bins


def bin_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


class BinSettings:
    """Container resolving and documenting all histogram bin widths for a run."""

    def __init__(
        self,
        mpp: Optional[float] = None,
        fps: Optional[float] = None,
        step_bin_width_um: Any = "auto",
        diffusion_bin_width: Any = "auto",
        logd_bin_width: Any = "auto",
        logmsd_bin_width: Any = "auto",
    ) -> None:
        self.mpp = mpp
        self.fps = fps
        step_floor = (float(mpp) / 10.0) if mpp else None
        self.step_width, self.step_source = resolve_bin_width(
            step_bin_width_um, default_value=0.01, min_reasonable=step_floor,
            label="Step-size")
        self.diffusion_width, self.diffusion_source = resolve_bin_width(
            diffusion_bin_width, default_value=0.01, label="Diffusion")
        self.logd_width, self.logd_source = resolve_bin_width(
            logd_bin_width, default_value=0.1, label="log10(D)")
        self.logmsd_width, self.logmsd_source = resolve_bin_width(
            logmsd_bin_width, default_value=0.1, label="log10(MSD)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mpp": self.mpp,
            "fps": self.fps,
            "step_bin_width_um": self.step_width,
            "step_bin_source": self.step_source,
            "step_floor_um": (float(self.mpp) / 10.0) if self.mpp else None,
            "diffusion_bin_width": self.diffusion_width,
            "diffusion_bin_source": self.diffusion_source,
            "logd_bin_width": self.logd_width,
            "logd_bin_source": self.logd_source,
            "logmsd_bin_width": self.logmsd_width,
            "logmsd_bin_source": self.logmsd_source,
        }
