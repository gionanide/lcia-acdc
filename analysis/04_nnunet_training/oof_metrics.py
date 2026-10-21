"""Native-grid metrics using official SimpleITK geometry and MONAI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.metrics import (
    compute_dice,
    compute_hausdorff_distance,
    compute_surface_dice,
)
from monai.networks.utils import one_hot

from training_io import assert_development_case, sha256_file

LABELS = {1: "rv", 2: "myocardium", 3: "lv"}
NSD_TOLERANCE_MM = 2.0


@dataclass(frozen=True)
class LabelVolume:
    labels: np.ndarray
    nibabel_affine: np.ndarray
    header_spacing: tuple[float, ...]
    axis_codes: tuple[str, ...]
    qform_code: int
    sform_code: int
    sform_axis_scales: tuple[float, ...]
    sform_scale_matches_pixdim: bool
    official_size: tuple[int, ...]
    official_spacing: tuple[float, ...]
    official_origin: tuple[float, ...]
    official_direction: tuple[float, ...]


def _simpleitk_geometry(
    path: Path,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    previous_warning_state = sitk.ProcessObject.GetGlobalWarningDisplay()
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    try:
        image = sitk.ReadImage(str(path.resolve()))
    finally:
        sitk.ProcessObject.SetGlobalWarningDisplay(previous_warning_state)
    return (
        tuple(int(value) for value in image.GetSize()),
        tuple(float(value) for value in image.GetSpacing()),
        tuple(float(value) for value in image.GetOrigin()),
        tuple(float(value) for value in image.GetDirection()),
    )


def _load_label(path: Path) -> LabelVolume:
    image = nib.load(str(path.resolve()))
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise FloatingPointError(f"Nonfinite segmentation values: {path}")
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        raise RuntimeError(f"Noninteger segmentation labels: {path}")
    labels = rounded.astype(np.int16, copy=False)
    unexpected = sorted(set(np.unique(labels).tolist()) - {0, 1, 2, 3})
    if unexpected:
        raise RuntimeError(f"Unexpected segmentation labels {unexpected}: {path}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[: labels.ndim])
    if len(spacing) != labels.ndim or not np.isfinite(spacing).all():
        raise RuntimeError(f"Invalid native spacing: {path}")
    _, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    if sform is None:
        sform_scales = tuple(float("nan") for _ in range(labels.ndim))
        sform_matches = False
    else:
        sform_scales = tuple(
            float(np.linalg.norm(sform[:3, axis])) for axis in range(labels.ndim)
        )
        sform_matches = bool(np.allclose(sform_scales, spacing, rtol=0, atol=1e-6))
    official_size, official_spacing, official_origin, official_direction = (
        _simpleitk_geometry(path)
    )
    return LabelVolume(
        labels=labels,
        nibabel_affine=np.asarray(image.affine, dtype=np.float64),
        header_spacing=spacing,
        axis_codes=tuple(str(value) for value in nib.aff2axcodes(image.affine)),
        qform_code=int(qform_code or 0),
        sform_code=int(sform_code or 0),
        sform_axis_scales=sform_scales,
        sform_scale_matches_pixdim=sform_matches,
        official_size=official_size,
        official_spacing=official_spacing,
        official_origin=official_origin,
        official_direction=official_direction,
    )


def _geometry_comparison(
    prediction: LabelVolume, reference: LabelVolume
) -> dict[str, Any]:
    shape_match = prediction.labels.shape == reference.labels.shape
    header_spacing_match = bool(
        np.allclose(
            prediction.header_spacing,
            reference.header_spacing,
            rtol=0,
            atol=1e-6,
        )
    )
    axis_codes_match = prediction.axis_codes == reference.axis_codes
    official_size_match = prediction.official_size == reference.official_size
    official_spacing_match = bool(
        np.allclose(
            prediction.official_spacing,
            reference.official_spacing,
            rtol=0,
            atol=1e-6,
        )
    )
    official_origin_match = bool(
        np.allclose(
            prediction.official_origin,
            reference.official_origin,
            rtol=0,
            atol=1e-5,
        )
    )
    official_direction_match = bool(
        np.allclose(
            prediction.official_direction,
            reference.official_direction,
            rtol=0,
            atol=1e-6,
        )
    )
    reference_header_matches_official = bool(
        np.allclose(
            reference.header_spacing,
            reference.official_spacing,
            rtol=0,
            atol=1e-6,
        )
    )
    prediction_header_matches_official = bool(
        np.allclose(
            prediction.header_spacing,
            prediction.official_spacing,
            rtol=0,
            atol=1e-6,
        )
    )
    official_geometry_match = all(
        (
            official_size_match,
            official_spacing_match,
            official_origin_match,
            official_direction_match,
        )
    )
    nibabel_affine_match = bool(
        np.allclose(
            prediction.nibabel_affine,
            reference.nibabel_affine,
            rtol=0,
            atol=1e-5,
        )
    )
    native_grid_match = all(
        (
            shape_match,
            header_spacing_match,
            axis_codes_match,
            official_geometry_match,
            reference_header_matches_official,
            prediction_header_matches_official,
        )
    )
    return {
        "native_shape_match": shape_match,
        "native_header_spacing_match": header_spacing_match,
        "native_axis_codes_match": axis_codes_match,
        "official_io_size_match": official_size_match,
        "official_io_spacing_match": official_spacing_match,
        "official_io_origin_match": official_origin_match,
        "official_io_direction_match": official_direction_match,
        "official_io_geometry_match": official_geometry_match,
        "reference_header_spacing_matches_official_io": reference_header_matches_official,
        "prediction_header_spacing_matches_official_io": prediction_header_matches_official,
        "nibabel_best_affine_match": nibabel_affine_match,
        "nibabel_best_affine_max_abs_difference": float(
            np.max(np.abs(prediction.nibabel_affine - reference.nibabel_affine))
        ),
        "nifti_header_encoding_differs": not nibabel_affine_match,
        "native_grid_match": native_grid_match,
    }


def _one_hot(labels: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))[None, None]
    return one_hot(tensor, num_classes=4, dim=1)


def _metric_values(tensor: torch.Tensor, name: str) -> list[float]:
    values = tensor.detach().cpu().numpy().reshape(-1)
    if len(values) != 3:
        raise RuntimeError(
            f"Expected three foreground {name} values, found {len(values)}"
        )
    return [float(value) for value in values]


def evaluate_case(
    prediction_path: Path,
    reference_path: Path,
    *,
    case_identifier: str,
    fold: int,
    checkpoint_rule: str,
) -> dict[str, Any]:
    patient_number, phase = assert_development_case(case_identifier)
    prediction = _load_label(prediction_path)
    reference = _load_label(reference_path)
    geometry = _geometry_comparison(prediction, reference)
    if not geometry["native_grid_match"]:
        raise RuntimeError(
            f"Native-grid geometry mismatch for {case_identifier}: {geometry}"
        )
    prediction_one_hot = _one_hot(prediction.labels)
    reference_one_hot = _one_hot(reference.labels)
    dice = _metric_values(
        compute_dice(
            prediction_one_hot,
            reference_one_hot,
            include_background=False,
            ignore_empty=True,
        ),
        "Dice",
    )
    hd95 = _metric_values(
        compute_hausdorff_distance(
            prediction_one_hot,
            reference_one_hot,
            include_background=False,
            distance_metric="euclidean",
            percentile=95,
            directed=False,
            spacing=[reference.official_spacing],
        ),
        "HD95",
    )
    nsd = _metric_values(
        compute_surface_dice(
            prediction_one_hot,
            reference_one_hot,
            class_thresholds=[NSD_TOLERANCE_MM] * 3,
            include_background=False,
            distance_metric="euclidean",
            spacing=[reference.official_spacing],
            use_subvoxels=True,
        ),
        "normalized surface Dice",
    )
    row: dict[str, Any] = {
        "patient": f"patient{patient_number:03d}",
        "patient_number": patient_number,
        "case_identifier": case_identifier,
        "phase": phase,
        "fold": fold,
        "checkpoint_rule": checkpoint_rule,
        "native_shape": "x".join(str(value) for value in reference.labels.shape),
        "native_spacing_mm": "x".join(
            f"{value:.8g}" for value in reference.official_spacing
        ),
        "native_axis_codes": "".join(reference.axis_codes),
        **geometry,
        "reference_qform_code": reference.qform_code,
        "reference_sform_code": reference.sform_code,
        "prediction_qform_code": prediction.qform_code,
        "prediction_sform_code": prediction.sform_code,
        "reference_sform_axis_scales": "x".join(
            f"{value:.8g}" for value in reference.sform_axis_scales
        ),
        "prediction_sform_axis_scales": "x".join(
            f"{value:.8g}" for value in prediction.sform_axis_scales
        ),
        "reference_sform_scale_matches_pixdim": reference.sform_scale_matches_pixdim,
        "prediction_sform_scale_matches_pixdim": prediction.sform_scale_matches_pixdim,
        "prediction_labels_valid": True,
        "prediction_sha256": sha256_file(prediction_path),
        "reference_sha256": sha256_file(reference_path),
        "nsd_tolerance_mm": NSD_TOLERANCE_MM,
        "nsd_use_subvoxels": True,
    }
    for index, label in LABELS.items():
        offset = index - 1
        row[f"{label}_dice"] = dice[offset]
        row[f"{label}_hd95_mm"] = hd95[offset]
        row[f"{label}_nsd_2mm"] = nsd[offset]
        row[f"{label}_reference_present"] = bool(np.any(reference.labels == index))
        row[f"{label}_prediction_present"] = bool(np.any(prediction.labels == index))
    metric_values = [
        row[f"{label}_{metric}"]
        for label in LABELS.values()
        for metric in ("dice", "hd95_mm", "nsd_2mm")
    ]
    row["finite_metrics"] = bool(np.isfinite(metric_values).all())
    row["valid_case"] = bool(
        row["finite_metrics"]
        and all(row[f"{label}_reference_present"] for label in LABELS.values())
        and all(row[f"{label}_prediction_present"] for label in LABELS.values())
    )
    return row


def evaluate_group(
    prediction_dir: Path,
    reference_dir: Path,
    *,
    case_identifiers: Iterable[str],
    fold: int,
    checkpoint_rule: str,
) -> pd.DataFrame:
    rows = []
    for identifier in case_identifiers:
        rows.append(
            evaluate_case(
                prediction_dir / f"{identifier}.nii.gz",
                reference_dir / f"{identifier}.nii.gz",
                case_identifier=identifier,
                fold=fold,
                checkpoint_rule=checkpoint_rule,
            )
        )
    frame = pd.DataFrame(rows).sort_values(["patient_number", "phase"])
    if frame["case_identifier"].duplicated().any():
        raise RuntimeError("Duplicate cases in segmentation metric group")
    return frame.reset_index(drop=True)


def aggregate_patients(case_metrics: pd.DataFrame) -> pd.DataFrame:
    required_rules = case_metrics["checkpoint_rule"].unique().tolist()
    if len(required_rules) != 1:
        raise RuntimeError("Patient aggregation requires exactly one checkpoint rule")
    rows: list[dict[str, Any]] = []
    for patient, group in case_metrics.groupby("patient", sort=True):
        if set(group["phase"]) != {"ED", "ES"} or len(group) != 2:
            raise RuntimeError(f"Patient {patient} does not have exactly ED and ES")
        by_phase = group.set_index("phase")
        row: dict[str, Any] = {
            "patient": patient,
            "patient_number": int(group["patient_number"].iloc[0]),
            "fold": int(group["fold"].iloc[0]),
            "checkpoint_rule": required_rules[0],
            "valid_patient": bool(group["valid_case"].all()),
        }
        for label in LABELS.values():
            for metric in ("dice", "hd95_mm", "nsd_2mm"):
                ed = float(by_phase.loc["ED", f"{label}_{metric}"])
                es = float(by_phase.loc["ES", f"{label}_{metric}"])
                row[f"{label}_{metric}_ed"] = ed
                row[f"{label}_{metric}_es"] = es
                row[f"{label}_{metric}_mean_phase"] = float(np.mean([ed, es]))
                if metric == "dice":
                    row[f"{label}_{metric}_minimum_phase"] = min(ed, es)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("patient_number").reset_index(drop=True)
