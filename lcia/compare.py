"""Level 1 of LCIA from two folders of ACDC-style label files.

Usage: python -m lcia.compare <original_dir> <refined_dir> <out.csv>
Each folder holds patientXXX/patientXXX_frameYY_gt.nii.gz (original ACDC layout) or
patientXXX_frameYY_gt.nii.gz files; ED and ES frames are read from the ACDC Info.cfg of the original folder.
"""
from __future__ import annotations
import sys, glob, os, re
import numpy as np, pandas as pd, nibabel as nib
from .measure import volume_ml, rvef, dice
from .envelope import convention_envelope, decision_flag


def _find(folder, pid, frame):
    pats = [f"{folder}/{pid}/{pid}_frame{frame:02d}_gt.nii.gz", f"{folder}/{pid}_frame{frame:02d}_gt.nii.gz"]
    for p in pats:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(pats[0])


def _frames(original_dir, pid):
    cfg = open(f"{original_dir}/{pid}/Info.cfg").read()
    ed = int(re.search(r"ED:\s*(\d+)", cfg).group(1)); es = int(re.search(r"ES:\s*(\d+)", cfg).group(1))
    return ed, es


def compare(original_dir, refined_dir, thresholds=(40, 45)):
    rows = []
    for d in sorted(glob.glob(f"{original_dir}/patient*")):
        pid = os.path.basename(d)
        ed, es = _frames(original_dir, pid)
        r = {"patient": pid}
        for name, fr in (("ed", ed), ("es", es)):
            o = nib.load(_find(original_dir, pid, fr)); f = nib.load(_find(refined_dir, pid, fr))
            vs = o.header.get_zooms()[:3]
            r[f"original_{name}v_ml"] = volume_ml(o.get_fdata(), vs)
            r[f"refined_{name}v_ml"] = volume_ml(f.get_fdata(), vs)
            r[f"dice_{name}"] = dice(o.get_fdata(), f.get_fdata())
        r["original_rvef"] = rvef(r["original_edv_ml"], r["original_esv_ml"])
        r["refined_rvef"] = rvef(r["refined_edv_ml"], r["refined_esv_ml"])
        lo, hi = convention_envelope(r["original_rvef"], r["refined_rvef"])
        r["envelope_low"], r["envelope_high"] = float(lo), float(hi)
        for t in thresholds:
            r[f"flag_{t}"] = int(decision_flag(r["original_rvef"], r["refined_rvef"], t))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2]).to_csv(sys.argv[3], index=False)
