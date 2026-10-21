#!/usr/bin/env python
"""Shared utilities for the ACDC annotation-convention monitor."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import torch
from scipy import stats
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_MODULES = PROJECT_ROOT / "analysis" / "02_deployed_models"
if str(MODEL_MODULES) not in sys.path:
    sys.path.insert(0, str(MODEL_MODULES))

import config as C  # noqa: E402
from dataset import (  # noqa: E402
    _crop_or_pad,
    _normalize,
    _resample_inplane,
    frame_file,
    list_patients,
    parse_info_cfg,
    stratified_split,
)
from model_zoo import build_model  # noqa: E402


RV_LABEL = int(getattr(C, "LBL_RV", 1))
MYO_LABEL = int(getattr(C, "LBL_MYO", 2))
LV_LABEL = int(getattr(C, "LBL_LV", 3))
NUM_CLASSES = int(getattr(C, "NUM_CLASSES", 4))
THRESHOLDS = (40.0, 45.0)
FAILURE_TOLERANCE = 5.0
ARCHITECTURES = ("unet", "segresnet", "attnunet", "swinunetr")
CONVENTIONS = ("original", "refined")
PHASES = ("ED", "ES")

PATIENT_RE = re.compile(r"(patient\d{3})", re.IGNORECASE)
FRAME_RE = re.compile(r"(?:frame|fr)[_\-]?(\d{1,3})", re.IGNORECASE)


@dataclass(frozen=True)
class LabelKey:
    patient: str
    frame: int


@dataclass(frozen=True)
class SliceRecord:
    patient: str
    group: str
    phase: str
    frame: int
    slice_index: int
    image_path: Path
    original_label_path: Path
    refined_label_path: Path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_int(*parts: Any, bits: int = 64) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, "little", signed=False)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def git_version(cwd: Path = PROJECT_ROOT) -> Dict[str, Any]:
    def run_git(args: List[str]) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception as exc:
            return f"unavailable: {exc}"
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip()
            return f"unavailable: {msg}"
        return proc.stdout.strip()

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }


def optional_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def package_versions() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "monai": optional_version("monai"),
        "nibabel": nib.__version__,
    }


def command_line() -> str:
    return " ".join([Path(sys.executable).name, *sys.argv])


def set_acdc_paths(acdc_root: Path, runs_root: Optional[Path] = None) -> None:
    C.DATA_ROOT = Path(acdc_root)
    C.TRAIN_DIR = C.DATA_ROOT / "training"
    C.TEST_DIR = C.DATA_ROOT / "testing"
    if runs_root is not None:
        C.OUT_DIR = Path(runs_root)


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def parse_refined_key(path: Path) -> Optional[LabelKey]:
    patient = PATIENT_RE.search(str(path))
    frame = FRAME_RE.search(path.name)
    if patient is None or frame is None:
        return None
    return LabelKey(patient.group(1).lower(), int(frame.group(1)))


def find_refined_labels(refined_root: Path) -> Dict[LabelKey, Path]:
    labels: Dict[LabelKey, Path] = {}
    for path in refined_root.rglob("*"):
        if not path.is_file() or not is_nifti(path):
            continue
        key = parse_refined_key(path)
        if key is None:
            continue
        if key in labels:
            raise RuntimeError(f"Duplicate refined label for {key}: {labels[key]} and {path}")
        labels[key] = path
    return labels


def all_acdc_patient_dirs(acdc_root: Path) -> List[Path]:
    train = [Path(p) for p in list_patients(Path(acdc_root) / "training")]
    test = [Path(p) for p in list_patients(Path(acdc_root) / "testing")]
    return sorted(train + test, key=lambda p: p.name)


def acdc_split(acdc_root: Path, split_seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    set_acdc_paths(acdc_root)
    train, val = stratified_split(C.TRAIN_DIR, C.VAL_FRACTION, split_seed)
    test = list_patients(C.TEST_DIR)
    return [Path(p) for p in train], [Path(p) for p in val], [Path(p) for p in test]


def patient_frame_paths(pdir: Path, refined_labels: Mapping[LabelKey, Path]) -> Dict[str, Dict[str, Any]]:
    pid = pdir.name.lower()
    info = parse_info_cfg(str(pdir))
    out: Dict[str, Dict[str, Any]] = {}
    for phase, frame_idx in (("ED", int(info["ED"])), ("ES", int(info["ES"]))):
        refined = refined_labels.get(LabelKey(pid, frame_idx))
        if refined is None:
            raise FileNotFoundError(f"Missing refined label for {pid} frame {frame_idx}")
        out[phase] = {
            "patient": pid,
            "group": info["Group"],
            "frame": frame_idx,
            "image": Path(frame_file(str(pdir), pid, frame_idx, gt=False)),
            "original": Path(frame_file(str(pdir), pid, frame_idx, gt=True)),
            "refined": refined,
        }
    return out


def load_native_frame(paths: Mapping[str, Any]) -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray, np.ndarray]:
    image_nii = nib.load(str(paths["image"]))
    original_nii = nib.load(str(paths["original"]))
    refined_nii = nib.load(str(paths["refined"]))
    image = np.asarray(image_nii.dataobj).astype(np.float32)
    original = np.rint(np.asarray(original_nii.dataobj)).astype(np.int16)
    refined = np.rint(np.asarray(refined_nii.dataobj)).astype(np.int16)
    if image.shape != original.shape or image.shape != refined.shape:
        raise RuntimeError(
            f"Shape mismatch for {paths['patient']} frame {paths['frame']}: "
            f"image {image.shape}, original {original.shape}, refined {refined.shape}"
        )
    if not np.allclose(original_nii.affine, refined_nii.affine, atol=1e-4, rtol=0):
        raise RuntimeError(f"Affine mismatch for {paths['patient']} frame {paths['frame']}")
    if original_nii.header.get_zooms()[:3] != refined_nii.header.get_zooms()[:3]:
        raise RuntimeError(f"Spacing mismatch for {paths['patient']} frame {paths['frame']}")
    return image_nii, image, original, refined


def target_spacing() -> Tuple[float, float]:
    ts = C.TARGET_SPACING
    if isinstance(ts, (tuple, list)):
        return float(ts[0]), float(ts[1])
    return float(ts), float(ts)


def preprocess_slice(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    image_resampled = _resample_inplane(image_slice, (sx, sy), C.TARGET_SPACING, is_mask=False)
    label_resampled = _resample_inplane(label_slice, (sx, sy), C.TARGET_SPACING, is_mask=True)
    resampled_shape = tuple(int(x) for x in image_resampled.shape)
    image = _normalize(_crop_or_pad(image_resampled, C.PATCH_SIZE, pad_value=0))
    label = _crop_or_pad(label_resampled, C.PATCH_SIZE, pad_value=0).astype(np.int16)
    return image, label, resampled_shape


def preprocess_volume(
    image_native: np.ndarray,
    label_native: np.ndarray,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    images = []
    labels = []
    shapes: List[Tuple[int, int]] = []
    for z in range(image_native.shape[2]):
        image, label, shape = preprocess_slice(image_native[:, :, z], label_native[:, :, z], sx, sy)
        images.append(image)
        labels.append(label)
        shapes.append(shape)
    return np.stack(images, axis=0), np.stack(labels, axis=0), shapes


def invert_crop_or_pad(patch: np.ndarray, resampled_shape: Tuple[int, int]) -> np.ndarray:
    h, w = resampled_shape
    th, tw = C.PATCH_SIZE
    sh = max(0, (h - th) // 2)
    sw = max(0, (w - tw) // 2)
    ch = min(h, th)
    cw = min(w, tw)
    dh = (th - ch) // 2
    dw = (tw - cw) // 2
    out = np.zeros((h, w), dtype=patch.dtype)
    out[sh : sh + ch, sw : sw + cw] = patch[dh : dh + ch, dw : dw + cw]
    return out


def resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.int16, copy=False)
    zoom_factors = (shape[0] / max(mask.shape[0], 1), shape[1] / max(mask.shape[1], 1))
    resized = zoom(mask, zoom_factors, order=0)
    return _crop_or_pad(resized, shape, pad_value=0).astype(np.int16)


def processed_prediction_to_native(
    pred_processed: np.ndarray,
    resampled_shapes: Sequence[Tuple[int, int]],
    native_shape: Tuple[int, int, int],
) -> np.ndarray:
    native_slices = []
    for z in range(pred_processed.shape[0]):
        uncropped = invert_crop_or_pad(pred_processed[z], resampled_shapes[z])
        native_slices.append(resize_mask_to_shape(uncropped, native_shape[:2]))
    return np.stack(native_slices, axis=2).astype(np.int16)


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool, copy=False)
    bb = b.astype(bool, copy=False)
    den = int(aa.sum()) + int(bb.sum())
    if den == 0:
        return float("nan")
    return float(2.0 * np.logical_and(aa, bb).sum() / den)


def volume_ml(mask: np.ndarray, voxel_ml: float) -> float:
    return float(np.count_nonzero(mask) * voxel_ml)


def ef(edv: float, esv: float) -> float:
    if not np.isfinite(edv) or edv <= 0:
        return float("nan")
    return float(100.0 * (edv - esv) / edv)


def rmse(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))


def interval_distance(value: float, a: float, b: float) -> float:
    if not np.isfinite([value, a, b]).all():
        return float("nan")
    lo, hi = sorted((float(a), float(b)))
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


def deterministic_augmented(
    image: np.ndarray,
    original_label: np.ndarray,
    refined_label: np.ndarray,
    seed: int,
    epoch: int,
    patient: str,
    phase: str,
    slice_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, bool]]:
    hflip = (stable_int(seed, epoch, patient, phase, slice_index, "hflip") % 2) == 1
    vflip = (stable_int(seed, epoch, patient, phase, slice_index, "vflip") % 2) == 1
    if hflip:
        image = image[:, ::-1]
        original_label = original_label[:, ::-1]
        refined_label = refined_label[:, ::-1]
    if vflip:
        image = image[::-1, :]
        original_label = original_label[::-1, :]
        refined_label = refined_label[::-1, :]
    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(original_label),
        np.ascontiguousarray(refined_label),
        {"hflip": bool(hflip), "vflip": bool(vflip)},
    )


def build_slice_records(patient_dirs: Sequence[Path], refined_labels: Mapping[LabelKey, Path]) -> List[SliceRecord]:
    records: List[SliceRecord] = []
    for pdir in sorted([Path(p) for p in patient_dirs], key=lambda p: p.name):
        frame_paths = patient_frame_paths(pdir, refined_labels)
        for phase in PHASES:
            paths = frame_paths[phase]
            n_slices = nib.load(str(paths["image"])).shape[2]
            for slice_index in range(n_slices):
                records.append(
                    SliceRecord(
                        patient=str(paths["patient"]),
                        group=str(paths["group"]),
                        phase=phase,
                        frame=int(paths["frame"]),
                        slice_index=int(slice_index),
                        image_path=Path(paths["image"]),
                        original_label_path=Path(paths["original"]),
                        refined_label_path=Path(paths["refined"]),
                    )
                )
    return records


def load_cached_training_slices(records: Sequence[SliceRecord]) -> Dict[str, Any]:
    images: List[np.ndarray] = []
    original_labels: List[np.ndarray] = []
    refined_labels: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    by_frame: Dict[Tuple[Path, Path, Path], Tuple[nib.Nifti1Image, np.ndarray, np.ndarray, np.ndarray]] = {}
    for rec in records:
        key = (rec.image_path, rec.original_label_path, rec.refined_label_path)
        if key not in by_frame:
            image_nii = nib.load(str(rec.image_path))
            original_nii = nib.load(str(rec.original_label_path))
            refined_nii = nib.load(str(rec.refined_label_path))
            image = np.asarray(image_nii.dataobj).astype(np.float32)
            original = np.rint(np.asarray(original_nii.dataobj)).astype(np.int16)
            refined = np.rint(np.asarray(refined_nii.dataobj)).astype(np.int16)
            if image.shape != original.shape or image.shape != refined.shape:
                raise RuntimeError(f"Shape mismatch while caching {rec.patient} {rec.phase}")
            by_frame[key] = (image_nii, image, original, refined)
        image_nii, image_native, original_native, refined_native = by_frame[key]
        sx, sy = [float(x) for x in image_nii.header.get_zooms()[:2]]
        image, original, _ = preprocess_slice(
            image_native[:, :, rec.slice_index],
            original_native[:, :, rec.slice_index],
            sx,
            sy,
        )
        _, refined, _ = preprocess_slice(
            image_native[:, :, rec.slice_index],
            refined_native[:, :, rec.slice_index],
            sx,
            sy,
        )
        images.append(image.astype(np.float32))
        original_labels.append(original.astype(np.int64))
        refined_labels.append(refined.astype(np.int64))
        meta.append(
            {
                "patient": rec.patient,
                "group": rec.group,
                "phase": rec.phase,
                "frame": rec.frame,
                "slice_index": rec.slice_index,
            }
        )
    return {
        "images": np.stack(images, axis=0),
        "original": np.stack(original_labels, axis=0),
        "refined": np.stack(refined_labels, axis=0),
        "meta": meta,
    }


def epoch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(stable_int(seed, epoch, "sample-order"))
    return rng.permutation(n)


def batch_from_cache(
    cache: Mapping[str, Any],
    indices: Sequence[int],
    seed: int,
    epoch: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_images = []
    batch_original = []
    batch_refined = []
    for idx in indices:
        m = cache["meta"][int(idx)]
        image, original, refined, _ = deterministic_augmented(
            cache["images"][int(idx)],
            cache["original"][int(idx)],
            cache["refined"][int(idx)],
            seed=seed,
            epoch=epoch,
            patient=m["patient"],
            phase=m["phase"],
            slice_index=int(m["slice_index"]),
        )
        batch_images.append(image)
        batch_original.append(original)
        batch_refined.append(refined)
    x = torch.from_numpy(np.stack(batch_images, axis=0)).unsqueeze(1).to(device)
    y_original = torch.from_numpy(np.stack(batch_original, axis=0)).unsqueeze(1).long().to(device)
    y_refined = torch.from_numpy(np.stack(batch_refined, axis=0)).unsqueeze(1).long().to(device)
    return x, y_original, y_refined


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_unet_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    if "best_by_ef" in path.name.lower() or "best_by_ef" in str(path.parent).lower():
        raise RuntimeError(f"Refusing to load selection-biased checkpoint: {path}")
    net = build_model("unet").to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return net


@torch.inference_mode()
def infer_processed(
    net: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_slices: int,
    return_probs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = []
    probs = [] if return_probs else None
    for start in range(0, len(images), batch_slices):
        x = torch.from_numpy(images[start : start + batch_slices]).unsqueeze(1).to(device)
        logits = net(x)
        if return_probs:
            p = torch.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy().astype(np.int16))
    pred = np.concatenate(preds, axis=0)
    prob = np.concatenate(probs, axis=0) if return_probs and probs is not None else None
    return pred, prob


def mean_entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    ent = -np.sum(clipped * np.log(clipped), axis=1)
    return float(ent.mean())


def basal_zone(slice_index: int, n_slices: int) -> str:
    if n_slices <= 1:
        return "middle"
    pos = slice_index / float(n_slices - 1)
    if pos < 1.0 / 3.0:
        return "basal"
    if pos < 2.0 / 3.0:
        return "middle"
    return "apical"


def binary_metrics(y: Iterable[int], score: Iterable[float]) -> Dict[str, float]:
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(score), dtype=float)
    valid = np.isfinite(s_arr)
    y_arr = y_arr[valid]
    s_arr = s_arr[valid]
    n = int(len(y_arr))
    n_pos = int(y_arr.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        auroc = float("nan")
    else:
        ranks = stats.rankdata(s_arr, method="average")
        sum_pos = float(ranks[y_arr == 1].sum())
        auroc = float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    if n_pos == 0:
        auprc = float("nan")
    else:
        order = np.argsort(-s_arr, kind="mergesort")
        y_sorted = y_arr[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / n_pos
        recall_prev = np.r_[0.0, recall[:-1]]
        auprc = float(np.sum((recall - recall_prev) * precision))
    if n < 3 or np.unique(y_arr).size < 2:
        spearman = float("nan")
    else:
        with np.errstate(invalid="ignore"):
            spearman = float(stats.spearmanr(s_arr, y_arr).statistic)
    return {
        "n": n,
        "n_events": n_pos,
        "event_rate": float(n_pos / n) if n else float("nan"),
        "spearman": spearman,
        "auroc": auroc,
        "auprc": auprc,
    }


def bootstrap_ci(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap(
    df: pd.DataFrame,
    patient_col: str,
    statistic_fn,
    n_boot: int,
    seed: int,
) -> Tuple[Any, Tuple[Any, Any], pd.DataFrame]:
    patients = np.asarray(sorted(df[patient_col].unique()))
    point = statistic_fn(df)
    rng = np.random.default_rng(seed)
    rows = []
    values = []
    for boot_id in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat(
            [df[df[patient_col] == pid] for pid in sampled],
            ignore_index=True,
        )
        value = statistic_fn(sample)
        rows.append({"bootstrap_id": boot_id, "value": value})
        values.append(value)
    lo, hi = bootstrap_ci(values)
    return point, (lo, hi), pd.DataFrame(rows)


def regression_stats(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[valid]
    yy = yy[valid]
    if len(xx) < 3:
        return {
            "pearson": float("nan"),
            "spearman": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
        }
    pearson = float(stats.pearsonr(xx, yy).statistic)
    spearman = float(stats.spearmanr(xx, yy).statistic)
    slope, intercept = np.polyfit(xx, yy, 1)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def risk_coverage_rows(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    review_rates: Sequence[float],
    seed: int,
    n_boot: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    patients = np.asarray(sorted(df["pid"].unique()))
    rng = np.random.default_rng(seed)

    def rates_for(sample: pd.DataFrame) -> Dict[float, float]:
        out = {}
        ordered = sample.sort_values(score_col, ascending=False)
        n = len(ordered)
        for rr in review_rates:
            n_review = int(math.floor(rr * n))
            accepted = ordered.iloc[n_review:]
            out[rr] = float(accepted[target_col].mean()) if len(accepted) else float("nan")
        return out

    point = rates_for(df)
    boot_values = {rr: [] for rr in review_rates}
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([df[df.pid == pid] for pid in sampled], ignore_index=True)
        values = rates_for(sample)
        for rr, value in values.items():
            boot_values[rr].append(value)
    for rr in review_rates:
        lo, hi = bootstrap_ci(boot_values[rr])
        rows.append(
            {
                "score": score_col,
                "target": target_col,
                "review_rate": rr,
                "coverage": 1.0 - rr,
                "accepted_event_rate": point[rr],
                "accepted_event_rate_ci_low": lo,
                "accepted_event_rate_ci_high": hi,
            }
        )
    return pd.DataFrame(rows)
