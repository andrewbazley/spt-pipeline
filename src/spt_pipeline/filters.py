"""Declarative track filtering, applied once and reported honestly.

Every figure in a results folder should be traceable to the exact population
behind it. That is the whole point of this module: filters are declared up
front, applied to one shared per-track table, and written out as a report
saying how many tracks each criterion removed and how many survived per
condition. A filtered analysis whose selection you cannot reconstruct is not a
result, it is an anecdote.

Filters compose as an AND. Each is a ``(min, max)`` window on a column, with
either end optional, so ``--filter-d 0.05 -`` reads as "D at least 0.05".

On order and attrition
----------------------
Criteria are applied in a fixed, declared order and the report shows the
attrition at each step, which makes it visible when two filters are largely
redundant (the second removes almost nothing) or when one is doing all the
work. It also reports the surviving count *per condition*, because a filter
that removes 5% of one condition and 60% of another has silently turned a
comparison of treatments into a comparison of selection effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Filter name -> (column, human label). The column may be absent from a given
#: dataset, in which case the filter is reported as inapplicable rather than
#: silently ignored.
CRITERIA: Dict[str, Tuple[str, str]] = {
    "d":          ("D_fit_um2_s",        "diffusion coefficient (µm²/s)"),
    "velocity":   ("v_path_um_s",        "mean path speed (µm/s)"),
    "brightness": ("mass_per_frame",     "track brightness (a.u.)"),
    "intensity":  ("marker_intensity",   "marker intensity (a.u.)"),
    "alpha":      ("alpha",              "anomalous exponent"),
    "nspots":     ("n_spots",            "localizations per track"),
    "step":       ("step_median_um_tau1", "median step at τ=1 (µm)"),
    "edge":       ("dist_to_edge_px",    "distance to frame edge (px)"),
    "coverage":   ("frac_disk_in_frame", "marker disk inside frame"),
    # Crowding. `density` is the biology-facing axis (how crowded the ER was);
    # `linksafety` is the QC-facing one (how much room the linker had).
    "density":    ("track_median_field_localizations_n",
                   "local density (localizations in the field radius)"),
    "densityum2": ("track_median_field_localizations_per_um2",
                   "local density (localizations per um^2)"),
    "celldensity": ("cell_median_field_localizations_n",
                    "density of the cell the track came from"),
    "celllinksafety": ("cell_link_safety_ratio",
                       "link-safety ratio of the cell the track came from"),
    "linksafety": ("link_safety_ratio",
                   "link-safety ratio (neighbour spacing / typical step)"),
    "ambiguity":  ("p_link_ambiguous", "probability of an ambiguous link"),
    "neighbors":  ("track_median_neighbors_R_n",
                   "competitors inside the linking search radius"),
}


def _num(v) -> Optional[float]:
    """Parse a bound; '-', '', 'none' and None all mean 'unbounded'."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "-", "none", "nan", "inf", "+inf"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if np.isfinite(f) else None


@dataclass(frozen=True)
class Window:
    lo: Optional[float] = None
    hi: Optional[float] = None

    def __bool__(self) -> bool:
        return self.lo is not None or self.hi is not None

    def describe(self) -> str:
        if self.lo is not None and self.hi is not None:
            return f"[{self.lo:g}, {self.hi:g}]"
        if self.lo is not None:
            return f">= {self.lo:g}"
        if self.hi is not None:
            return f"<= {self.hi:g}"
        return "any"


@dataclass
class TrackFilter:
    """A set of windows, applied as an AND across track properties."""
    windows: Dict[str, Window] = field(default_factory=dict)
    #: Drop tracks whose value is missing for an active criterion. Off by
    #: default: a NaN means "not measured", and silently discarding unmeasured
    #: tracks would make a filter look far more selective than it is.
    drop_missing: bool = False

    @classmethod
    def from_args(cls, args, prefix: str = "filter_") -> "TrackFilter":
        """Build from argparse values named ``filter_<name>`` = (lo, hi)."""
        w: Dict[str, Window] = {}
        for name in CRITERIA:
            raw = getattr(args, f"{prefix}{name}", None)
            if not raw:
                continue
            lo, hi = (list(raw) + [None, None])[:2]
            win = Window(_num(lo), _num(hi))
            if win:
                w[name] = win
        return cls(windows=w, drop_missing=bool(getattr(args, "filter_drop_missing", False)))

    def __bool__(self) -> bool:
        return any(bool(w) for w in self.windows.values())

    def describe(self) -> Dict[str, str]:
        return {name: w.describe() for name, w in self.windows.items() if w}

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter ``df``; return ``(kept, report)``.

        The report has one row per criterion in application order, plus a
        ``TOTAL`` row, recording how many tracks entered, how many survived,
        and how many were removed by that criterion alone.
        """
        rows = []
        n_start = len(df)
        out = df
        for name, win in self.windows.items():
            if not win:
                continue
            col, label = CRITERIA[name]
            before = len(out)
            if col not in out.columns:
                rows.append({"criterion": name, "column": col, "window": win.describe(),
                             "status": "column not present -- NOT APPLIED",
                             "n_before": before, "n_after": before, "n_removed": 0})
                continue
            v = pd.to_numeric(out[col], errors="coerce")
            keep = pd.Series(True, index=out.index)
            if win.lo is not None:
                keep &= (v >= win.lo)
            if win.hi is not None:
                keep &= (v <= win.hi)
            n_missing = int(v.isna().sum())
            keep = keep.fillna(False) if self.drop_missing else (keep | v.isna())
            out = out[keep]
            rows.append({"criterion": name, "column": col, "window": win.describe(),
                         "status": (f"{n_missing} missing "
                                    f"{'dropped' if self.drop_missing else 'kept'}")
                                   if n_missing else "ok",
                         "n_before": before, "n_after": len(out),
                         "n_removed": before - len(out)})
        rows.append({"criterion": "TOTAL", "column": "", "window": "",
                     "status": "", "n_before": n_start, "n_after": len(out),
                     "n_removed": n_start - len(out)})
        return out, pd.DataFrame(rows)


def per_condition_attrition(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Survival per condition, and per movie, after filtering.

    Uneven attrition across conditions is the failure mode that matters: it
    converts a comparison of treatments into a comparison of selection
    effects. This table makes that visible instead of leaving it implicit.
    """
    if before.empty or "condition" not in before.columns:
        return pd.DataFrame()
    b = before.groupby("condition").size().rename("n_before")
    a = (after.groupby("condition").size().rename("n_after")
         if not after.empty else pd.Series(dtype=int, name="n_after"))
    out = pd.concat([b, a], axis=1).fillna(0).astype(int).reset_index()
    out["frac_kept"] = np.where(out["n_before"] > 0,
                                out["n_after"] / out["n_before"], np.nan)
    if "movie_id" in before.columns:
        mb = before.groupby("condition")["movie_id"].nunique().rename("movies_before")
        ma = (after.groupby("condition")["movie_id"].nunique().rename("movies_after")
              if not after.empty else pd.Series(dtype=int, name="movies_after"))
        out = out.merge(pd.concat([mb, ma], axis=1).fillna(0).astype(int).reset_index(),
                        on="condition", how="left")
    return out


def absolute_bins(values: pd.Series, edges: Optional[Sequence[float]] = None,
                  n_quantile: Optional[int] = None,
                  prefix: str = "B") -> Tuple[pd.Series, pd.DataFrame]:
    """Bin a column, either on absolute edges or on quantiles.

    Absolute edges keep bin boundaries identical across conditions, which is
    what "compare tracks of the same brightness" requires -- quantile bins
    would put a dim track from one condition in the same bin as a bright track
    from another, which is the opposite of matching.

    Returns ``(labels, edge_table)``; labels are NaN outside the outermost
    edges, so absolute binning also acts as a filter on the tails.
    """
    v = pd.to_numeric(values, errors="coerce")
    if edges:
        e = np.asarray(sorted(float(x) for x in edges), dtype=float)
        kind = "absolute"
    elif n_quantile and n_quantile >= 2:
        e = np.unique(np.nanquantile(v, np.linspace(0, 1, int(n_quantile) + 1)))
        kind = "quantile"
    else:
        return pd.Series(np.nan, index=values.index), pd.DataFrame()
    if e.size < 3:
        return pd.Series(np.nan, index=values.index), pd.DataFrame()
    labels = [f"{prefix}{i + 1}" for i in range(e.size - 1)]
    cut = pd.cut(v, bins=e, labels=labels, include_lowest=True)
    table = pd.DataFrame({"bin": labels, "lower": e[:-1], "upper": e[1:],
                          "kind": kind})
    table["n"] = [int((cut == l).sum()) for l in labels]
    return cut, table
