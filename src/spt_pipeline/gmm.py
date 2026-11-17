"""BIC-selected Gaussian mixture models for D subpopulation decomposition.

Fits mixtures of 1..K Gaussians (typically to log10 D) and selects the number
of components by minimum BIC. Used to characterize mobility subpopulations
(e.g. confined vs mobile GEMs). One definition here replaces the copies that
lived in every legacy script.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False


def bic_select(values: np.ndarray, max_k: int = 5, random_state: int = 42):
    """Fit GMMs for K=1..max_k and return (best_model, bic_list).

    ``values`` is a 1D array (NaNs/inf removed internally).
    """
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required for GMM fitting.")
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)].reshape(-1, 1)
    if len(vals) < 2:
        return None, []
    max_k = max(1, min(int(max_k), len(vals)))
    bics: List[float] = []
    models = []
    for k in range(1, max_k + 1):
        g = GaussianMixture(n_components=k, random_state=random_state, n_init=5).fit(vals)
        bics.append(float(g.bic(vals)))
        models.append(g)
    best = int(np.argmin(bics))
    return models[best], bics


def subpopulation_table(model, log_scale: bool = True) -> pd.DataFrame:
    """Tidy component table (weight, mean, std) sorted by mean.

    When ``log_scale`` the means/stds were fit on log10(D); a linear-space
    center (10**mean) is also reported for interpretability.
    """
    if model is None:
        return pd.DataFrame(columns=["component", "weight", "mean", "std", "center_linear"])
    means = model.means_.flatten()
    stds = np.sqrt(model.covariances_.flatten())
    weights = model.weights_.flatten()
    order = np.argsort(means)
    rows = []
    for rank, idx in enumerate(order):
        rows.append({
            "component": rank,
            "weight": float(weights[idx]),
            "mean": float(means[idx]),
            "std": float(stds[idx]),
            "center_linear": float(10 ** means[idx]) if log_scale else float(means[idx]),
        })
    return pd.DataFrame(rows)


def fit_log_d(diff_df: pd.DataFrame, d_col: str = "D_fit_um2_s", max_k: int = 5
              ) -> Tuple[Optional[object], List[float], pd.DataFrame]:
    """Convenience: fit GMM to log10(D) from a diffusion table."""
    if diff_df is None or diff_df.empty or d_col not in diff_df.columns:
        return None, [], pd.DataFrame()
    d = pd.to_numeric(diff_df[d_col], errors="coerce").to_numpy()
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 2:
        return None, [], pd.DataFrame()
    logd = np.log10(d)
    model, bics = bic_select(logd, max_k=max_k)
    return model, bics, subpopulation_table(model, log_scale=True)
