"""Volumes, RVEF and Dice from ACDC-style label maps (RV label value 1)."""
from __future__ import annotations
import numpy as np

RV = 1


def volume_ml(label_map, voxel_size_mm, label=RV):
    """Volume of `label` in mL from the voxel count and the voxel size (mm)."""
    return float((np.asarray(label_map) == label).sum() * np.prod(voxel_size_mm) / 1000.0)


def rvef(edv_ml, esv_ml):
    """Ejection fraction in percent, 100 (EDV - ESV) / EDV."""
    return 100.0 * (edv_ml - esv_ml) / edv_ml


def dice(label_a, label_b, label=RV):
    """Dice between the masks of `label` in two label maps (1.0 if both are empty)."""
    a = np.asarray(label_a) == label
    b = np.asarray(label_b) == label
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)
