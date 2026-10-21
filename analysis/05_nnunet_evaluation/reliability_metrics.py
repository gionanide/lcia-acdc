"""Validated native measurements, MONAI segmentation metrics, and patient statistics."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
    compute_surface_dice,
)
from monai.networks.utils import one_hot
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.proportion import proportion_confint


LABELS = {1: "rv", 2: "myocardium", 3: "lv"}
NSD_TOLERANCE_MM = 2.0
NEAR_ZERO_EDV_ML = 1.0
SCALED_LABEL_ABSOLUTE_TOLERANCE = 1e-6


def canonicalize_scaled_labels(
    values: np.ndarray, *, atol: float = SCALED_LABEL_ABSOLUTE_TOLERANCE
) -> tuple[np.ndarray, float]:
    """Canonicalize NIfTI-scaled class codes only within a frozen tiny tolerance."""
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite scaled label values")
    rounded = np.rint(array)
    maximum_deviation = float(np.max(np.abs(array - rounded))) if array.size else 0.0
    if maximum_deviation > atol:
        raise ValueError(
            f"Scaled label deviation {maximum_deviation:.12g} exceeds frozen tolerance {atol}"
        )
    labels = rounded.astype(np.uint8, copy=False)
    unexpected = set(np.unique(labels).tolist()) - {0, 1, 2, 3}
    if unexpected:
        raise ValueError(f"Unexpected canonical label values: {sorted(unexpected)}")
    return labels, maximum_deviation


def remap_mnms_to_instrument(labels: np.ndarray) -> np.ndarray:
    """Map M&Ms 0/BG,1/LV,2/MYO,3/RV to 0/BG,1/RV,2/MYO,3/LV."""
    values = np.asarray(labels)
    unexpected = set(np.unique(values).tolist()) - {0, 1, 2, 3}
    if unexpected:
        raise ValueError(f"Unexpected M&Ms labels: {sorted(unexpected)}")
    result = values.astype(np.uint8, copy=True)
    result[values == 1] = 3
    result[values == 3] = 1
    return result


def native_volume_ml(labels: np.ndarray, spacing_mm: Sequence[float], label: int = 1) -> float:
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Invalid native spacing: {spacing_mm}")
    voxel_ml = float(np.prod(spacing) / 1000.0)
    return float(np.count_nonzero(np.asarray(labels) == label) * voxel_ml)


def rvef(edv_ml: float, esv_ml: float) -> float:
    if not np.isfinite(edv_ml) or not np.isfinite(esv_ml) or edv_ml <= 0:
        return float("nan")
    return float(100.0 * (edv_ml - esv_ml) / edv_ml)


def measurement_from_phase_volumes(edv_ml: float, esv_ml: float) -> dict[str, float]:
    return {
        "edv_ml": float(edv_ml),
        "esv_ml": float(esv_ml),
        "stroke_volume_ml": float(edv_ml - esv_ml),
        "rvef_points": rvef(edv_ml, esv_ml),
    }


def pathology_flags(edv_ml: float, esv_ml: float, rvef_points: float) -> dict[str, bool]:
    finite_edv = bool(np.isfinite(edv_ml))
    finite_esv = bool(np.isfinite(esv_ml))
    finite_ef = bool(np.isfinite(rvef_points))
    return {
        "nonfinite_measurement": not (finite_edv and finite_esv and finite_ef),
        "catastrophic_denominator_failure": (not finite_edv) or edv_ml <= 0.0,
        "near_zero_predicted_edv": finite_edv and edv_ml < NEAR_ZERO_EDV_ML,
        "predicted_esv_gt_edv": finite_edv and finite_esv and esv_ml > edv_ml,
        "finite_rvef_outside_0_100": finite_ef and (rvef_points < 0.0 or rvef_points > 100.0),
    }


def interval_distance(value: float, first: float, second: float) -> float:
    if not np.isfinite([value, first, second]).all():
        return float("nan")
    lower, upper = sorted((float(first), float(second)))
    return float(max(lower - value, 0.0, value - upper))


def signed_interval_error(value: float, first: float, second: float) -> float:
    if not np.isfinite([value, first, second]).all():
        return float("nan")
    lower, upper = sorted((float(first), float(second)))
    if value < lower:
        return float(value - lower)
    if value > upper:
        return float(value - upper)
    return 0.0


def category_disagreement(predicted: float, reference: float, threshold: float) -> float:
    if not np.isfinite([predicted, reference]).all():
        return float("nan")
    return float((predicted < threshold) != (reference < threshold))


def _read_label(path: Path) -> tuple[sitk.Image, np.ndarray]:
    previous = sitk.ProcessObject.GetGlobalWarningDisplay()
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    try:
        image = sitk.ReadImage(str(path.resolve()))
    finally:
        sitk.ProcessObject.SetGlobalWarningDisplay(previous)
    data = sitk.GetArrayFromImage(image)
    if not np.isfinite(data).all():
        raise FloatingPointError(f"Nonfinite segmentation values: {path}")
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        raise RuntimeError(f"Noninteger segmentation values: {path}")
    labels = rounded.astype(np.int16, copy=False)
    unexpected = set(np.unique(labels).tolist()) - {0, 1, 2, 3}
    if unexpected:
        raise RuntimeError(f"Unexpected segmentation labels {sorted(unexpected)}: {path}")
    return image, labels


def geometry_record(image: sitk.Image) -> dict[str, Any]:
    return {
        "size": tuple(int(value) for value in image.GetSize()),
        "spacing": tuple(float(value) for value in image.GetSpacing()),
        "origin": tuple(float(value) for value in image.GetOrigin()),
        "direction": tuple(float(value) for value in image.GetDirection()),
    }


def geometry_matches(first: sitk.Image, second: sitk.Image) -> bool:
    a, b = geometry_record(first), geometry_record(second)
    return bool(
        a["size"] == b["size"]
        and np.allclose(a["spacing"], b["spacing"], rtol=0, atol=1e-6)
        and np.allclose(a["origin"], b["origin"], rtol=0, atol=1e-5)
        and np.allclose(a["direction"], b["direction"], rtol=0, atol=1e-6)
    )


def _one_hot(labels: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))[None, None]
    return one_hot(tensor, num_classes=4, dim=1)


def _three_values(values: torch.Tensor, name: str) -> list[float]:
    array = values.detach().cpu().numpy().reshape(-1)
    if len(array) != 3:
        raise RuntimeError(f"Expected three foreground {name} values, got {len(array)}")
    return [float(value) for value in array]


def segmentation_case_metrics(prediction_path: Path, reference_path: Path) -> dict[str, Any]:
    prediction_image, prediction = _read_label(prediction_path)
    reference_image, reference = _read_label(reference_path)
    if not geometry_matches(prediction_image, reference_image):
        raise RuntimeError(
            f"Native-grid mismatch: prediction={geometry_record(prediction_image)}, "
            f"reference={geometry_record(reference_image)}"
        )
    prediction_hot = _one_hot(prediction)
    reference_hot = _one_hot(reference)
    # SimpleITK arrays are z,y,x, so spacing follows the same tensor-axis order.
    spacing_zyx = [tuple(reversed(reference_image.GetSpacing()))]
    dice = _three_values(
        compute_dice(prediction_hot, reference_hot, include_background=False, ignore_empty=True),
        "Dice",
    )
    hd95 = _three_values(
        compute_hausdorff_distance(
            prediction_hot,
            reference_hot,
            include_background=False,
            distance_metric="euclidean",
            percentile=95,
            directed=False,
            spacing=spacing_zyx,
        ),
        "HD95",
    )
    nsd = _three_values(
        compute_surface_dice(
            prediction_hot,
            reference_hot,
            class_thresholds=[NSD_TOLERANCE_MM] * 3,
            include_background=False,
            distance_metric="euclidean",
            spacing=spacing_zyx,
            use_subvoxels=True,
        ),
        "NSD",
    )
    assd = _three_values(
        compute_average_surface_distance(
            prediction_hot,
            reference_hot,
            include_background=False,
            symmetric=True,
            distance_metric="euclidean",
            spacing=spacing_zyx,
        ),
        "ASSD",
    )
    row: dict[str, Any] = {
        "native_grid_match": True,
        "native_size": "x".join(str(v) for v in reference_image.GetSize()),
        "native_spacing_mm": "x".join(f"{v:.8g}" for v in reference_image.GetSpacing()),
        "nsd_tolerance_mm": NSD_TOLERANCE_MM,
    }
    for index, label in LABELS.items():
        offset = index - 1
        row.update(
            {
                f"{label}_dice": dice[offset],
                f"{label}_hd95_mm": hd95[offset],
                f"{label}_nsd_2mm": nsd[offset],
                f"{label}_assd_mm": assd[offset],
                f"{label}_reference_present": bool(np.any(reference == index)),
                f"{label}_prediction_present": bool(np.any(prediction == index)),
                f"{label}_reference_support_slices": int(np.any(reference == index, axis=(1, 2)).sum()),
                f"{label}_prediction_support_slices": int(np.any(prediction == index, axis=(1, 2)).sum()),
            }
        )
    return row


def phase_volume_ml(path: Path, label: int = 1) -> float:
    image, labels = _read_label(path)
    return native_volume_ml(labels, image.GetSpacing(), label=label)


def aggregate_patient_segmentation(case_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["cohort", "patient_id", "vendor", "pathology", "model", "reference_policy"]
    for key, group in case_rows.groupby(keys, dropna=False, sort=True):
        if set(group["phase"]) != {"ED", "ES"} or len(group) != 2:
            raise RuntimeError(f"Incomplete ED/ES segmentation pair: {key}")
        by_phase = group.set_index("phase")
        row = dict(zip(keys, key))
        for label in LABELS.values():
            for metric in ("dice", "hd95_mm", "nsd_2mm", "assd_mm"):
                ed = float(by_phase.loc["ED", f"{label}_{metric}"])
                es = float(by_phase.loc["ES", f"{label}_{metric}"])
                row[f"{label}_{metric}_ed"] = ed
                row[f"{label}_{metric}_es"] = es
                row[f"{label}_{metric}_mean_phase"] = float(np.mean([ed, es]))
                if metric in {"dice", "nsd_2mm"}:
                    row[f"{label}_{metric}_minimum_phase"] = float(min(ed, es))
                else:
                    row[f"{label}_{metric}_maximum_phase"] = float(max(ed, es))
        rows.append(row)
    return pd.DataFrame(rows)


def continuous_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "n_total": int(array.size),
            "n_valid": 0,
            "n_invalid": int(array.size),
            "mean": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "percentile_5": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "n_total": int(array.size),
        "n_valid": int(finite.size),
        "n_invalid": int(array.size - finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q1": float(np.quantile(finite, 0.25)),
        "q3": float(np.quantile(finite, 0.75)),
        "percentile_5": float(np.quantile(finite, 0.05)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def bca_interval(
    values: Iterable[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 5000,
    seed: int = 20260812,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    estimate = float(statistic(array))
    if array.size == 1 or np.allclose(array, array[0], rtol=0, atol=0):
        return {"estimate": estimate, "ci_low": estimate, "ci_high": estimate, "n": int(array.size)}
    result = stats.bootstrap(
        (array,),
        statistic,
        vectorized=False,
        paired=False,
        confidence_level=0.95,
        n_resamples=n_resamples,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    return {
        "estimate": estimate,
        "ci_low": float(result.confidence_interval.low),
        "ci_high": float(result.confidence_interval.high),
        "n": int(array.size),
    }


def wilson_interval(events: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"events": int(events), "total": int(total), "rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    low, high = proportion_confint(events, total, alpha=0.05, method="wilson")
    return {
        "events": int(events),
        "total": int(total),
        "rate": float(events / total),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def error_battery(predicted: Iterable[float], reference: Iterable[float]) -> dict[str, Any]:
    pred = np.asarray(list(predicted), dtype=np.float64)
    ref = np.asarray(list(reference), dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError("Prediction/reference measurement shape mismatch")
    valid = np.isfinite(pred) & np.isfinite(ref)
    error = pred[valid] - ref[valid]
    if error.size == 0:
        return {"n_total": int(pred.size), "n_valid": 0, "n_invalid": int(pred.size)}
    sd = float(np.std(error, ddof=1)) if error.size > 1 else float("nan")
    mean = float(np.mean(error))
    return {
        "n_total": int(pred.size),
        "n_valid": int(error.size),
        "n_invalid": int(pred.size - error.size),
        "signed_bias": mean,
        "mae": float(np.mean(np.abs(error))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "error_sd_ddof1": sd,
        "bland_altman_lower_1_96sd": float(mean - 1.96 * sd),
        "bland_altman_upper_1_96sd": float(mean + 1.96 * sd),
    }


def mean_pairwise_absolute(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if len(finite) < 2:
        return float("nan")
    return float(np.mean([abs(a - b) for a, b in itertools.combinations(finite, 2)]))


def discrimination_metrics(labels: Iterable[float], scores: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.float64)
    score = np.asarray(list(scores), dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(score)
    y = y[valid].astype(int)
    score = score[valid]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return {
            "n": int(len(y)),
            "positives": positives,
            "negatives": negatives,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
        }
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))

    def statistic(y_sample: np.ndarray, score_sample: np.ndarray) -> float:
        labels_here = y_sample.astype(int)
        if len(np.unique(labels_here)) != 2:
            return float("nan")
        return float(roc_auc_score(labels_here, score_sample))

    ci_low = ci_high = float("nan")
    if positives >= 10 and negatives >= 10:
        result = stats.bootstrap(
            (y, score),
            statistic,
            paired=True,
            vectorized=False,
            confidence_level=0.95,
            n_resamples=5000,
            method="BCa",
            random_state=np.random.default_rng(20260812),
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)
    return {
        "n": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "auroc": auroc,
        "auprc": auprc,
        "auroc_ci_low": ci_low,
        "auroc_ci_high": ci_high,
    }
