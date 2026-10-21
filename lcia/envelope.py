"""Convention envelope, decision-stability flag and envelope error (Eqs. 1-3 of the paper)."""
from __future__ import annotations
import numpy as np


def convention_envelope(rvef_original, rvef_refined):
    """Return (lower, upper) bounds of the RVEF interval spanned by the two label sets (Eq. 1)."""
    a = np.asarray(rvef_original, dtype=float)
    b = np.asarray(rvef_refined, dtype=float)
    return np.minimum(a, b), np.maximum(a, b)


def decision_flag(rvef_original, rvef_refined, tau):
    """Decision-stability flag f_tau (Eq. 2): 1 if the two label sets place the RVEF on different sides of tau."""
    lo, hi = convention_envelope(rvef_original, rvef_refined)
    return ((lo < tau) & (tau <= hi)).astype(int)


def envelope_error(rvef_predicted, rvef_original, rvef_refined):
    """Distance of a predicted RVEF from the convention envelope (Eq. 3); zero inside the envelope."""
    y = np.asarray(rvef_predicted, dtype=float)
    a = np.asarray(rvef_original, dtype=float)
    b = np.asarray(rvef_refined, dtype=float)
    lo, hi = convention_envelope(a, b)
    inside = (y >= lo) & (y <= hi)
    return np.where(inside, 0.0, np.minimum(np.abs(y - a), np.abs(y - b)))
