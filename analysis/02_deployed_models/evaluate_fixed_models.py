#!/usr/bin/env python
"""
Reference-fragility experiment for ACDC cardiac segmentation.

Purpose
-------
Run the same fixed historical model predictions against two expert-supported
ACDC RV reference conventions (original and refined), then quantify how the
reference choice changes:
  * reference EDV/ESV/RVEF,
  * Dice and model RVEF error,
  * 3/5/10-point failure labels,
  * high-overlap silent-failure labels,
  * model ranking,
  * reference-aware error (distance to the interval spanned by both references).

This script intentionally evaluates only ACDC patients 101-150 for model claims.
The full 150-patient reference-only comparison is a separate analysis step.

Predicted measurements use the processed grid; references use native grids.
Dice is computed on the processed grid. Historical epoch counts are unproved.

Example (PowerShell)
--------------------
python evaluate_fixed_models.py `
  --acdc-dir "ACDC" `
  --refined-dir "RefinedLabelsACDC" `
  --runs-root ".\\runs" `
  --out-dir ".\\refined_acdc\\model_reference_audit"

Requirements
------------
numpy, pandas, scipy, nibabel, torch, MONAI
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau
import torch

import config as C
from dataset import (
    _crop_or_pad,
    _normalize,
    _resample_inplane,
    frame_file,
    list_patients,
    parse_info_cfg,
)
from model_zoo import ARCHS, build_model


PATIENT_RE = re.compile(r"(patient\d{3})", re.IGNORECASE)
FRAME_RE = re.compile(r"(?:frame|fr)[_\-]?(\d{1,3})", re.IGNORECASE)
RV_LABEL = int(getattr(C, "LBL_RV", 1))
THRESHOLDS = (3.0, 5.0, 10.0)
KNOWN_UNET_PROCESSED_GATE = {
    "ef_bias": -0.53,
    "ef_mae": 5.12,
    "dice_mean": 0.8939,
}


@dataclass(frozen=True)
class LabelKey:
    patient: str
    frame: int


def is_nifti(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def parse_key(path: Path) -> Optional[LabelKey]:
    pm = PATIENT_RE.search(str(path))
    fm = FRAME_RE.search(path.name)
    if pm is None or fm is None:
        return None
    return LabelKey(pm.group(1).lower(), int(fm.group(1)))


def find_refined_labels(root: Path) -> Dict[LabelKey, Path]:
    labels: Dict[LabelKey, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or not is_nifti(path):
            continue
        key = parse_key(path)
        if key is None:
            continue
        if key in labels:
            raise RuntimeError(
                f"Duplicate refined label for {key}: {labels[key]} and {path}"
            )
        labels[key] = path
    return labels


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def resolve_checkpoint(runs_root: Path, arch: str) -> Path:
    if arch == "unet":
        candidates = [runs_root / "last.pt", runs_root / "last_unet.pt"]
    else:
        candidates = [runs_root / f"last_{arch}.pt"]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No fixed last-named checkpoint found for {arch}. Tried: "
            + ", ".join(str(p) for p in candidates)
        )
    path = existing[0]
    if "best_by_ef" in path.name.lower():
        raise RuntimeError("Refusing to use a selection-biased best_by_ef checkpoint.")
    return path


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool, copy=False)
    g = gt.astype(bool, copy=False)
    den = int(p.sum()) + int(g.sum())
    if den == 0:
        return float("nan")
    return float(2.0 * np.logical_and(p, g).sum() / den)


def ef(edv: float, esv: float) -> float:
    if not np.isfinite(edv) or edv <= 0:
        return float("nan")
    return float(100.0 * (edv - esv) / edv)


def rmse(x: pd.Series) -> float:
    a = x.to_numpy(dtype=float)
    return float(np.sqrt(np.nanmean(a * a)))


def reference_interval_distance(pred: float, a: float, b: float) -> float:
    if not all(np.isfinite([pred, a, b])):
        return float("nan")
    lo, hi = sorted((a, b))
    if pred < lo:
        return float(lo - pred)
    if pred > hi:
        return float(pred - hi)
    return 0.0


def target_spacing() -> Tuple[float, float]:
    ts = C.TARGET_SPACING
    if isinstance(ts, (tuple, list)):
        return float(ts[0]), float(ts[1])
    return float(ts), float(ts)


def preprocess_volume(
    image_native: np.ndarray,
    label_native: np.ndarray,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for z in range(image_native.shape[2]):
        image = _resample_inplane(
            image_native[:, :, z], (sx, sy), C.TARGET_SPACING, is_mask=False
        )
        label = _resample_inplane(
            label_native[:, :, z], (sx, sy), C.TARGET_SPACING, is_mask=True
        )
        image = _normalize(_crop_or_pad(image, C.PATCH_SIZE, pad_value=0))
        label = _crop_or_pad(label, C.PATCH_SIZE, pad_value=0).astype(np.int16)
        images.append(image)
        labels.append(label)
    return np.stack(images, axis=0), np.stack(labels, axis=0)


@torch.inference_mode()
def infer_slices(
    net: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_slices: int,
) -> np.ndarray:
    preds = []
    for start in range(0, len(images), batch_slices):
        x = torch.from_numpy(images[start : start + batch_slices]).unsqueeze(1).to(device)
        logits = net(x)
        preds.append(logits.argmax(1).cpu().numpy().astype(np.int16))
    return np.concatenate(preds, axis=0)


def volume_ml(mask: np.ndarray, voxel_ml: float) -> float:
    return float(np.count_nonzero(mask) * voxel_ml)


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    n_boot: int,
    statistic=np.mean,
) -> Tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(a))
    if n_boot <= 0 or len(a) == 1:
        return point, float("nan"), float("nan")
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        out[i] = statistic(rng.choice(a, size=len(a), replace=True))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    rng: np.random.Generator,
    n_boot: int,
    statistic=np.mean,
) -> Tuple[float, float, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa, bb = aa[valid], bb[valid]
    if len(aa) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(bb) - statistic(aa))
    if n_boot <= 0 or len(aa) == 1:
        return point, float("nan"), float("nan")
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(aa), size=len(aa))
        out[i] = statistic(bb[idx]) - statistic(aa[idx])
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def format_ci(x: Tuple[float, float, float], digits: int = 3) -> str:
    p, lo, hi = x
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def collect_case(
    net: torch.nn.Module,
    arch: str,
    ckpt: Path,
    ckpt_hash: str,
    pdir: Path,
    refined: Dict[LabelKey, Path],
    device: torch.device,
    batch_slices: int,
) -> dict:
    pid = pdir.name.lower()
    info = parse_info_cfg(str(pdir))
    row: dict = {
        "pid": pid,
        "group": info.get("Group", "?"),
        "arch": arch,
        "checkpoint": ckpt.name,
        "checkpoint_sha256": ckpt_hash,
    }

    tsx, tsy = target_spacing()

    for phase, frame_idx in (("ED", int(info["ED"])), ("ES", int(info["ES"]))):
        image_path = Path(frame_file(str(pdir), pid, frame_idx, gt=False))
        original_path = Path(frame_file(str(pdir), pid, frame_idx, gt=True))
        refined_path = refined.get(LabelKey(pid, frame_idx))
        if refined_path is None:
            raise FileNotFoundError(f"Missing refined label for {pid}, frame {frame_idx}")

        image_nii = nib.load(str(image_path))
        original_nii = nib.load(str(original_path))
        refined_nii = nib.load(str(refined_path))
        image_native = np.asarray(image_nii.dataobj).astype(np.float32)
        original_native = np.rint(np.asarray(original_nii.dataobj)).astype(np.int16)
        refined_native = np.rint(np.asarray(refined_nii.dataobj)).astype(np.int16)

        if image_native.shape != original_native.shape or original_native.shape != refined_native.shape:
            raise RuntimeError(
                f"Shape mismatch for {pid} {phase}: image {image_native.shape}, "
                f"original {original_native.shape}, refined {refined_native.shape}"
            )
        if not np.allclose(original_nii.affine, refined_nii.affine, atol=1e-4, rtol=0):
            raise RuntimeError(f"Affine mismatch for {pid} {phase}")

        sx, sy, sz = [float(z) for z in image_nii.header.get_zooms()[:3]]
        native_voxel_ml = sx * sy * sz / 1000.0
        processed_voxel_ml = tsx * tsy * sz / 1000.0

        images, original_processed = preprocess_volume(
            image_native, original_native, sx, sy
        )
        # The image preprocessing is identical for both references; process only
        # the refined mask on the second pass and discard the duplicate images.
        _, refined_processed = preprocess_volume(
            image_native, refined_native, sx, sy
        )
        pred = infer_slices(net, images, device, batch_slices)

        p_rv = pred == RV_LABEL
        o_rv_native = original_native == RV_LABEL
        r_rv_native = refined_native == RV_LABEL
        o_rv_proc = original_processed == RV_LABEL
        r_rv_proc = refined_processed == RV_LABEL

        row[f"pred_{phase}_ml"] = volume_ml(p_rv, processed_voxel_ml)
        row[f"orig_native_{phase}_ml"] = volume_ml(o_rv_native, native_voxel_ml)
        row[f"refined_native_{phase}_ml"] = volume_ml(r_rv_native, native_voxel_ml)
        row[f"orig_proc_{phase}_ml"] = volume_ml(o_rv_proc, processed_voxel_ml)
        row[f"refined_proc_{phase}_ml"] = volume_ml(r_rv_proc, processed_voxel_ml)
        row[f"dice_orig_{phase}"] = dice_binary(p_rv, o_rv_proc)
        row[f"dice_refined_{phase}"] = dice_binary(p_rv, r_rv_proc)
        row[f"ref_changed_voxels_{phase}"] = int(
            np.count_nonzero(original_native != refined_native)
        )
        row[f"rv_added_voxels_{phase}"] = int(
            np.count_nonzero(r_rv_native & ~o_rv_native)
        )
        row[f"rv_removed_voxels_{phase}"] = int(
            np.count_nonzero(o_rv_native & ~r_rv_native)
        )

    row["pred_rvef"] = ef(row["pred_ED_ml"], row["pred_ES_ml"])
    for reference in ("orig", "refined"):
        for convention in ("native", "proc"):
            ref_ef = ef(
                row[f"{reference}_{convention}_ED_ml"],
                row[f"{reference}_{convention}_ES_ml"],
            )
            row[f"{reference}_{convention}_rvef"] = ref_ef
            err = row["pred_rvef"] - ref_ef
            row[f"err_{reference}_{convention}"] = err
            row[f"abs_err_{reference}_{convention}"] = abs(err)

        d_ed = row[f"dice_{reference}_ED"]
        d_es = row[f"dice_{reference}_ES"]
        row[f"dice_{reference}_mean"] = float(np.nanmean([d_ed, d_es]))
        row[f"dice_{reference}_min"] = float(np.nanmin([d_ed, d_es]))

    row["delta_reference_rvef_native"] = (
        row["refined_native_rvef"] - row["orig_native_rvef"]
    )
    row["delta_reference_rvef_proc"] = (
        row["refined_proc_rvef"] - row["orig_proc_rvef"]
    )
    row["reference_spread_native"] = abs(row["delta_reference_rvef_native"])
    row["reference_spread_proc"] = abs(row["delta_reference_rvef_proc"])
    row["reference_aware_error_native"] = reference_interval_distance(
        row["pred_rvef"], row["orig_native_rvef"], row["refined_native_rvef"]
    )
    row["reference_aware_error_proc"] = reference_interval_distance(
        row["pred_rvef"], row["orig_proc_rvef"], row["refined_proc_rvef"]
    )

    for tau in THRESHOLDS:
        tag = str(int(tau))
        for convention in ("native", "proc"):
            fo = row[f"abs_err_orig_{convention}"] > tau
            fr = row[f"abs_err_refined_{convention}"] > tau
            row[f"fail_orig_{convention}_{tag}"] = int(fo)
            row[f"fail_refined_{convention}_{tag}"] = int(fr)
            row[f"failure_flip_{convention}_{tag}"] = int(fo != fr)
            row[f"robust_fail_{convention}_{tag}"] = int(
                row[f"reference_aware_error_{convention}"] > tau
            )
        for gate in ("mean", "min"):
            row[f"silent_orig_{gate}_{tag}"] = int(
                row[f"dice_orig_{gate}"] >= 0.90
                and row["abs_err_orig_native"] > tau
            )
            row[f"silent_refined_{gate}_{tag}"] = int(
                row[f"dice_refined_{gate}"] >= 0.90
                and row["abs_err_refined_native"] > tau
            )
    return row


def summarise_arch(
    g: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    arch = str(g.arch.iloc[0])
    out: dict = {"arch": arch, "n": int(len(g))}

    for ref in ("orig", "refined"):
        for convention in ("native", "proc"):
            e = g[f"err_{ref}_{convention}"]
            out[f"{ref}_{convention}_bias"] = float(e.mean())
            out[f"{ref}_{convention}_mae"] = float(e.abs().mean())
            out[f"{ref}_{convention}_median_ae"] = float(e.abs().median())
            out[f"{ref}_{convention}_rmse"] = rmse(e)
            out[f"{ref}_{convention}_error_sd"] = float(e.std(ddof=1))
        out[f"{ref}_dice_mean"] = float(g[f"dice_{ref}_mean"].mean())
        out[f"{ref}_dice_min"] = float(g[f"dice_{ref}_min"].mean())

    out["reference_aware_native_mae"] = float(g.reference_aware_error_native.mean())
    out["reference_aware_native_median"] = float(g.reference_aware_error_native.median())
    out["reference_aware_proc_mae"] = float(g.reference_aware_error_proc.mean())
    out["mean_reference_delta_native"] = float(g.delta_reference_rvef_native.mean())
    out["median_reference_spread_native"] = float(g.reference_spread_native.median())

    for tau in THRESHOLDS:
        tag = str(int(tau))
        for convention in ("native", "proc"):
            out[f"failure_flip_{convention}_{tag}"] = float(
                g[f"failure_flip_{convention}_{tag}"].mean()
            )
            out[f"robust_failure_{convention}_{tag}"] = float(
                g[f"robust_fail_{convention}_{tag}"].mean()
            )
        for ref in ("orig", "refined"):
            out[f"fail_{ref}_native_{tag}"] = float(
                g[f"fail_{ref}_native_{tag}"].mean()
            )
            for gate in ("mean", "min"):
                high = g[f"dice_{ref}_{gate}"] >= 0.90
                out[f"high_coverage_{ref}_{gate}"] = float(high.mean())
                out[f"silent_rate_{ref}_{gate}_{tag}"] = (
                    float(g.loc[high, f"abs_err_{ref}_native"].gt(tau).mean())
                    if high.any()
                    else float("nan")
                )
                out[f"silent_n_{ref}_{gate}_{tag}"] = int(
                    g.loc[high, f"abs_err_{ref}_native"].gt(tau).sum()
                )
                out[f"high_n_{ref}_{gate}"] = int(high.sum())

    # Patient-paired bootstrap for the change in MAE caused by the reference.
    out["delta_mae_refined_minus_orig_native"] = float(
        g.abs_err_refined_native.mean() - g.abs_err_orig_native.mean()
    )
    p, lo, hi = paired_bootstrap_difference(
        g.abs_err_orig_native,
        g.abs_err_refined_native,
        rng,
        n_boot,
        statistic=np.mean,
    )
    out["delta_mae_ci_low"] = lo
    out["delta_mae_ci_high"] = hi
    p, lo, hi = bootstrap_ci(
        g.failure_flip_native_5,
        rng,
        n_boot,
        statistic=np.mean,
    )
    out["failure_flip_native_5_ci_low"] = lo
    out["failure_flip_native_5_ci_high"] = hi
    return out


def make_rankings(summary: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    r = summary[[
        "arch",
        "orig_native_mae",
        "refined_native_mae",
        "reference_aware_native_mae",
        "orig_dice_mean",
        "refined_dice_mean",
    ]].copy()
    r["rank_mae_original"] = r.orig_native_mae.rank(method="min", ascending=True)
    r["rank_mae_refined"] = r.refined_native_mae.rank(method="min", ascending=True)
    r["rank_mae_reference_aware"] = r.reference_aware_native_mae.rank(
        method="min", ascending=True
    )
    r["rank_dice_original"] = r.orig_dice_mean.rank(method="min", ascending=False)
    r["rank_dice_refined"] = r.refined_dice_mean.rank(method="min", ascending=False)

    stats = {}
    pairs = [
        ("mae_original_vs_refined", "rank_mae_original", "rank_mae_refined"),
        ("dice_original_vs_refined", "rank_dice_original", "rank_dice_refined"),
        ("mae_original_vs_reference_aware", "rank_mae_original", "rank_mae_reference_aware"),
    ]
    for name, a, b in pairs:
        tau = kendalltau(r[a], r[b], nan_policy="omit")
        stats[name] = {"kendall_tau": float(tau.statistic), "p": float(tau.pvalue)}
    return r, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc-dir", "--acdc-root", dest="acdc_root", type=Path, required=True)
    ap.add_argument("--refined-dir", "--refined-root", dest="refined_root", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--architectures",
        nargs="+",
        default=list(ARCHS),
        choices=list(ARCHS),
    )
    ap.add_argument("--batch-slices", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()

    acdc_root = args.acdc_root.resolve()
    refined_root = args.refined_root.resolve()
    runs_root = args.runs_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override config paths for this process without editing config.py.
    C.DATA_ROOT = acdc_root
    C.TRAIN_DIR = acdc_root / "training"
    C.TEST_DIR = acdc_root / "testing"
    C.OUT_DIR = runs_root

    refined = find_refined_labels(refined_root)
    if len(refined) != 300:
        raise RuntimeError(
            f"Expected 300 refined ED/ES labels, found {len(refined)} under {refined_root}"
        )

    test_patients = [Path(p) for p in list_patients(C.TEST_DIR)]
    if len(test_patients) != 50:
        raise RuntimeError(f"Expected 50 ACDC test patients, found {len(test_patients)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device}  patients={len(test_patients)}  architectures={args.architectures}")

    all_rows = []
    checkpoint_manifest = []
    for arch in args.architectures:
        ckpt = resolve_checkpoint(runs_root, arch)
        ckpt_hash = sha256(ckpt)
        checkpoint_manifest.append(
            {"arch": arch, "path": str(ckpt), "sha256": ckpt_hash}
        )
        print(f"\n[{arch}] loading {ckpt}")
        net = build_model(arch).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        net.eval()

        for i, pdir in enumerate(test_patients, start=1):
            all_rows.append(
                collect_case(
                    net,
                    arch,
                    ckpt,
                    ckpt_hash,
                    pdir,
                    refined,
                    device,
                    args.batch_slices,
                )
            )
            if i % 10 == 0 or i == len(test_patients):
                print(f"  {i}/{len(test_patients)}")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cases = pd.DataFrame(all_rows).sort_values(["arch", "pid"]).reset_index(drop=True)
    cases_path = out_dir / "reference_fragility_cases.csv"
    cases.to_csv(cases_path, index=False)

    rng = np.random.default_rng(args.seed)
    summaries = pd.DataFrame(
        [summarise_arch(g, rng, args.n_boot) for _, g in cases.groupby("arch", sort=False)]
    )
    summary_path = out_dir / "reference_fragility_summary.csv"
    summaries.to_csv(summary_path, index=False)

    rankings, ranking_stats = make_rankings(summaries)
    rankings_path = out_dir / "reference_fragility_rankings.csv"
    rankings.to_csv(rankings_path, index=False)

    # Reference-only summary on the held-out test subset (one copy per patient).
    ref_test = cases[cases.arch == cases.arch.iloc[0]].copy()
    ref_summary = {
        "n_test_patients": int(len(ref_test)),
        "median_abs_reference_delta_native": float(ref_test.reference_spread_native.median()),
        "iqr_abs_reference_delta_native": [
            float(ref_test.reference_spread_native.quantile(0.25)),
            float(ref_test.reference_spread_native.quantile(0.75)),
        ],
        "mean_signed_reference_delta_native": float(ref_test.delta_reference_rvef_native.mean()),
        "n_abs_delta_gt_3": int((ref_test.reference_spread_native > 3).sum()),
        "n_abs_delta_gt_5": int((ref_test.reference_spread_native > 5).sum()),
        "n_abs_delta_gt_10": int((ref_test.reference_spread_native > 10).sum()),
        "max_abs_reference_delta_native": float(ref_test.reference_spread_native.max()),
        "native_vs_processed_delta_max_abs_difference": float(
            (ref_test.delta_reference_rvef_native - ref_test.delta_reference_rvef_proc).abs().max()
        ),
    }

    gate = None
    if "unet" in set(cases.arch):
        u = cases[cases.arch == "unet"]
        observed = {
            "ef_bias": float(u.err_orig_proc.mean()),
            "ef_mae": float(u.abs_err_orig_proc.mean()),
            "dice_mean": float(u.dice_orig_mean.mean()),
        }
        gate = {
            "expected": KNOWN_UNET_PROCESSED_GATE,
            "observed": observed,
            "absolute_difference": {
                k: abs(observed[k] - KNOWN_UNET_PROCESSED_GATE[k])
                for k in observed
            },
        }
        targets = {"native_rvef_bias": -0.54, "native_rvef_mae": 5.17,
                   "processed_rv_dice": 0.8939}
        tolerances = {"native_rvef_bias": 0.75, "native_rvef_mae": 0.50,
                      "processed_rv_dice": 0.01}
        observed = {"native_rvef_bias": float(u.err_orig_native.mean()),
                    "native_rvef_mae": float(u.abs_err_orig_native.mean()),
                    "processed_rv_dice": float(u.dice_orig_mean.mean())}
        reproduction = {"targets": targets, "tolerances": tolerances,
                        "observed": observed,
                        "pass": all(abs(observed[key] - targets[key]) <= tolerances[key]
                                    for key in targets),
                        "measurement_note": "Processed prediction volumes versus native references; processed Dice"}
        (out_dir / "reproduction_gate.json").write_text(
            json.dumps({"reproduction_gate": reproduction}, indent=2), encoding="utf-8")

    manifest = {
        "study": "ACDC original-vs-refined reference fragility",
        "created_by": Path(__file__).name,
        "acdc_root": str(acdc_root),
        "refined_root": str(refined_root),
        "runs_root": str(runs_root),
        "out_dir": str(out_dir),
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "architectures": args.architectures,
        "checkpoint_manifest": checkpoint_manifest,
        "thresholds_ef_points": list(THRESHOLDS),
        "primary_measurement_reference": "native-grid original/refined masks",
        "overlap_reference": "processed-grid masks using training preprocessing",
        "primary_high_overlap_gate": "min(ED Dice, ES Dice) >= 0.90",
        "secondary_high_overlap_gate": "mean(ED Dice, ES Dice) >= 0.90",
        "reference_test_summary": ref_summary,
        "ranking_stability": ranking_stats,
        "unet_reproduction_gate": gate,
        "outputs": {
            "cases": str(cases_path),
            "summary": str(summary_path),
            "rankings": str(rankings_path),
        },
    }
    manifest_path = out_dir / "reference_fragility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("REFERENCE-ONLY EFFECT ON ACDC TEST PATIENTS")
    print("=" * 88)
    print(json.dumps(ref_summary, indent=2))

    print("\n" + "=" * 88)
    print("MODEL SUMMARY")
    print("=" * 88)
    show = [
        "arch",
        "orig_native_mae",
        "refined_native_mae",
        "reference_aware_native_mae",
        "failure_flip_native_5",
        "orig_dice_mean",
        "refined_dice_mean",
        "silent_rate_orig_min_5",
        "silent_rate_refined_min_5",
    ]
    print(summaries[show].round(4).to_string(index=False))

    print("\n" + "=" * 88)
    print("RANKINGS")
    print("=" * 88)
    print(rankings.round(4).to_string(index=False))
    print(json.dumps(ranking_stats, indent=2))

    if gate is not None:
        print("\n" + "=" * 88)
        print("U-NET ORIGINAL-REFERENCE REPRODUCTION GATE (processed convention)")
        print("=" * 88)
        print(json.dumps(gate, indent=2))
        print(
            "Interpretation: small differences are expected only from software/version details. "
            "Large differences mean this pipeline is not reproducing the existing last.pt analysis."
        )

    print("\nWrote:")
    for p in (cases_path, summary_path, rankings_path, manifest_path):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
