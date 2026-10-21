#!/usr/bin/env python
"""nnU-Net evaluation Methods: cohort inputs and five folds to reliability tables.

One entry point covers input validation, official native inference and analysis.
No training, adaptation or checkpoint selection is performed here.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import reliability_common as common
from reliability_data import load_case_manifest
from reliability_inference import (
    C2_ROOT,
    FOLD_DISAGREEMENT_PATH,
    HISTORICAL_ROOT,
    INFERENCE_MANIFEST_PATH,
)
from reliability_metrics import (
    aggregate_patient_segmentation,
    bca_interval,
    category_disagreement,
    continuous_summary,
    discrimination_metrics,
    error_battery,
    interval_distance,
    measurement_from_phase_volumes,
    pathology_flags,
    phase_volume_ml,
    segmentation_case_metrics,
    signed_interval_error,
    wilson_interval,
)


CASE_OUTPUT = common.OUTPUT_DIR / "reliability_per_case.csv"
PATIENT_OUTPUT = common.OUTPUT_DIR / "reliability_per_patient.csv"
SEGMENTATION_BATTERY = common.OUTPUT_DIR / "segmentation_battery.csv"
MEASUREMENT_BATTERY = common.OUTPUT_DIR / "measurement_battery.csv"
REFERENCE_POLICY = common.OUTPUT_DIR / "reference_policy_analysis.csv"
REFERENCE_SUMMARY = common.OUTPUT_DIR / "reference_policy_summary.csv"
HISTORICAL_COMPARISON = common.OUTPUT_DIR / "historical_paired_comparison.csv"
FOLD_MONITOR = common.OUTPUT_DIR / "fold_monitor_summary.csv"
RISK_COVERAGE = common.OUTPUT_DIR / "risk_coverage.csv"
DECISION_PATH = common.OUTPUT_DIR / "reliability_decision.json"
REPORT_PATH = common.OUTPUT_DIR / "RELIABILITY_REPORT.md"
ANALYSIS_MANIFEST_PATH = common.OUTPUT_DIR / "reliability_analysis_manifest.json"

FIGURES = {
    "overlap": common.OUTPUT_DIR / "figure_overlap_vs_measurement_error.png",
    "reference": common.OUTPUT_DIR / "figure_reference_policy_sensitivity.png",
    "historical": common.OUTPUT_DIR / "figure_historical_comparison.png",
    "vendor": common.OUTPUT_DIR / "figure_mnms_vendor.png",
    "risk": common.OUTPUT_DIR / "figure_fold_risk_coverage.png",
}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else common.WORKSPACE / path


def _prediction_path(model: str, identifier: str) -> Path:
    if model == "c2_ensemble":
        return C2_ROOT / "ensemble" / f"{identifier}.nii.gz"
    if model == "historical_unet":
        return HISTORICAL_ROOT / f"{identifier}.nii.gz"
    raise ValueError(model)


def _reference_specs(row: dict[str, Any]) -> list[tuple[str, Path]]:
    if row["cohort"] == "ACDC_test":
        return [
            ("original", _resolve(str(row["original_reference_path"]))),
            ("refined", _resolve(str(row["refined_reference_path"]))),
        ]
    return [("mnms", _resolve(str(row["original_reference_path"])))]


def _case_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(cases) * 2 + int((cases["cohort"] == "ACDC_test").sum()) * 2
    completed = 0
    for case in cases.to_dict("records"):
        for model in ("c2_ensemble", "historical_unet"):
            prediction = _prediction_path(model, str(case["identifier"]))
            for reference_policy, reference in _reference_specs(case):
                metric = segmentation_case_metrics(prediction, reference)
                rows.append(
                    {
                        "cohort": case["cohort"],
                        "patient_id": case["patient_id"],
                        "patient_number": case["patient_number"],
                        "vendor": case["vendor"],
                        "vendor_name": case["vendor_name"],
                        "pathology": case["pathology"],
                        "phase": case["phase"],
                        "identifier": case["identifier"],
                        "model": model,
                        "reference_policy": reference_policy,
                        **metric,
                    }
                )
                completed += 1
                if completed % 100 == 0:
                    print(f"Segmentation metrics: {completed}/{total}", flush=True)
    return pd.DataFrame(rows).sort_values(
        ["cohort", "model", "reference_policy", "patient_id", "phase"]
    )


def _patient_metrics(cases: pd.DataFrame, case_metrics: pd.DataFrame) -> pd.DataFrame:
    segmentation = aggregate_patient_segmentation(case_metrics)
    seg_lookup = segmentation.set_index(
        ["cohort", "patient_id", "model", "reference_policy"]
    )
    volume_cache: dict[str, float] = {}

    def volume(path: Path) -> float:
        key = str(path.resolve())
        if key not in volume_cache:
            volume_cache[key] = phase_volume_ml(path)
        return volume_cache[key]

    rows: list[dict[str, Any]] = []
    for (cohort, patient_id), group in cases.groupby(["cohort", "patient_id"], sort=True):
        by_phase = group.set_index("phase")
        if set(by_phase.index) != {"ED", "ES"}:
            raise RuntimeError(f"Incomplete patient pair: {cohort}/{patient_id}")
        first = group.iloc[0]
        for model in ("c2_ensemble", "historical_unet"):
            pred_ed = _prediction_path(model, str(by_phase.loc["ED", "identifier"]))
            pred_es = _prediction_path(model, str(by_phase.loc["ES", "identifier"]))
            predicted = measurement_from_phase_volumes(volume(pred_ed), volume(pred_es))
            flags = pathology_flags(
                predicted["edv_ml"], predicted["esv_ml"], predicted["rvef_points"]
            )
            policies = ("original", "refined") if cohort == "ACDC_test" else ("mnms",)
            for policy in policies:
                column = (
                    f"{policy}_reference_path"
                    if policy in {"original", "refined"}
                    else "original_reference_path"
                )
                ref_ed = _resolve(str(by_phase.loc["ED", column]))
                ref_es = _resolve(str(by_phase.loc["ES", column]))
                reference = measurement_from_phase_volumes(volume(ref_ed), volume(ref_es))
                row: dict[str, Any] = {
                    "cohort": cohort,
                    "patient_id": patient_id,
                    "patient_number": first["patient_number"],
                    "vendor": first["vendor"],
                    "vendor_name": first["vendor_name"],
                    "pathology": first["pathology"],
                    "model": model,
                    "reference_policy": policy,
                    **{f"predicted_{name}": value for name, value in predicted.items()},
                    **{f"reference_{name}": value for name, value in reference.items()},
                    **flags,
                }
                for endpoint in ("edv_ml", "esv_ml", "stroke_volume_ml", "rvef_points"):
                    error = predicted[endpoint] - reference[endpoint]
                    row[f"error_{endpoint}"] = error
                    row[f"absolute_error_{endpoint}"] = abs(error) if np.isfinite(error) else float("nan")
                for threshold in (5.0, 10.0):
                    suffix = int(threshold)
                    value = row["absolute_error_rvef_points"]
                    row[f"rvef_error_gt{suffix}"] = (
                        float(value > threshold) if np.isfinite(value) else float("nan")
                    )
                for threshold in (40.0, 45.0):
                    suffix = int(threshold)
                    row[f"category_disagreement_{suffix}"] = category_disagreement(
                        predicted["rvef_points"], reference["rvef_points"], threshold
                    )
                seg = seg_lookup.loc[(cohort, patient_id, model, policy)].to_dict()
                row.update(seg)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cohort", "model", "reference_policy", "patient_id"]
    )


def _reference_policy_table(patients: pd.DataFrame) -> pd.DataFrame:
    acdc = patients[patients["cohort"] == "ACDC_test"]
    rows: list[dict[str, Any]] = []
    for (model, patient_id), group in acdc.groupby(["model", "patient_id"], sort=True):
        if set(group["reference_policy"]) != {"original", "refined"} or len(group) != 2:
            raise RuntimeError(f"Incomplete ACDC reference-policy pair: {model}/{patient_id}")
        by_policy = group.set_index("reference_policy")
        original = by_policy.loc["original"]
        refined = by_policy.loc["refined"]
        row: dict[str, Any] = {
            "cohort": "ACDC_test",
            "patient_id": patient_id,
            "patient_number": original["patient_number"],
            "pathology": original["pathology"],
            "model": model,
            "minimum_phase_rv_dice_original": original["rv_dice_minimum_phase"],
            "minimum_phase_rv_dice_refined": refined["rv_dice_minimum_phase"],
            "minimum_phase_rv_dice_both_references": min(
                original["rv_dice_minimum_phase"], refined["rv_dice_minimum_phase"]
            ),
        }
        for endpoint in ("edv_ml", "esv_ml", "stroke_volume_ml", "rvef_points"):
            predicted = float(original[f"predicted_{endpoint}"])
            first = float(original[f"reference_{endpoint}"])
            second = float(refined[f"reference_{endpoint}"])
            row[f"predicted_{endpoint}"] = predicted
            row[f"original_reference_{endpoint}"] = first
            row[f"refined_reference_{endpoint}"] = second
            row[f"reference_shift_{endpoint}"] = second - first
            row[f"reference_spread_{endpoint}"] = abs(second - first)
            row[f"robust_absolute_error_{endpoint}"] = interval_distance(predicted, first, second)
            row[f"robust_signed_error_{endpoint}"] = signed_interval_error(predicted, first, second)
        for threshold in (5.0, 10.0):
            suffix = int(threshold)
            original_failure = float(original["absolute_error_rvef_points"]) > threshold
            refined_failure = float(refined["absolute_error_rvef_points"]) > threshold
            robust_error = float(row["robust_absolute_error_rvef_points"])
            row[f"failure_flip_gt{suffix}"] = bool(original_failure != refined_failure)
            row[f"robust_rvef_error_gt{suffix}"] = bool(robust_error > threshold)
        for threshold in (40.0, 45.0):
            suffix = int(threshold)
            original_ref = float(original["reference_rvef_points"])
            refined_ref = float(refined["reference_rvef_points"])
            predicted = float(original["predicted_rvef_points"])
            row[f"reference_category_flip_{suffix}"] = bool(
                (original_ref < threshold) != (refined_ref < threshold)
            )
            row[f"model_disagreement_original_{suffix}"] = bool(
                (predicted < threshold) != (original_ref < threshold)
            )
            row[f"model_disagreement_refined_{suffix}"] = bool(
                (predicted < threshold) != (refined_ref < threshold)
            )
            row[f"model_disagreement_any_policy_{suffix}"] = bool(
                row[f"model_disagreement_original_{suffix}"]
                or row[f"model_disagreement_refined_{suffix}"]
            )
        row.update(
            pathology_flags(
                row["predicted_edv_ml"], row["predicted_esv_ml"], row["predicted_rvef_points"]
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "patient_id"])


def _strata(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    result = [("pooled", frame)]
    if frame["cohort"].iloc[0] == "MnMs":
        result.extend((f"vendor_{vendor}", group) for vendor, group in frame.groupby("vendor", sort=True))
    return result


def _segmentation_battery(patients: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "rv_dice_ed",
        "rv_dice_es",
        "rv_dice_mean_phase",
        "rv_dice_minimum_phase",
        "rv_nsd_2mm_mean_phase",
        "rv_nsd_2mm_minimum_phase",
        "rv_hd95_mm_maximum_phase",
        "rv_assd_mm_mean_phase",
        "myocardium_dice_mean_phase",
        "lv_dice_mean_phase",
    )
    rows: list[dict[str, Any]] = []
    for (cohort, model, policy), group in patients.groupby(
        ["cohort", "model", "reference_policy"], sort=True
    ):
        for stratum, subset in _strata(group):
            for metric in metrics:
                summary = continuous_summary(subset[metric])
                interval = bca_interval(subset[metric], np.mean)
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "reference_policy": policy,
                        "stratum": stratum,
                        "metric": metric,
                        **summary,
                        "mean_bca95_low": interval["ci_low"],
                        "mean_bca95_high": interval["ci_high"],
                    }
                )
    return pd.DataFrame(rows)


def _measurement_battery(patients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoints = ("edv_ml", "esv_ml", "stroke_volume_ml", "rvef_points")
    for (cohort, model, policy), group in patients.groupby(
        ["cohort", "model", "reference_policy"], sort=True
    ):
        for stratum, subset in _strata(group):
            for endpoint in endpoints:
                battery = error_battery(
                    subset[f"predicted_{endpoint}"], subset[f"reference_{endpoint}"]
                )
                errors = subset[f"error_{endpoint}"].to_numpy(dtype=float)
                absolute = subset[f"absolute_error_{endpoint}"].to_numpy(dtype=float)
                signed_ci = bca_interval(errors, np.mean)
                mae_ci = bca_interval(absolute, np.mean)
                row: dict[str, Any] = {
                    "cohort": cohort,
                    "model": model,
                    "reference_policy": policy,
                    "stratum": stratum,
                    "endpoint": endpoint,
                    **battery,
                    "signed_bias_bca95_low": signed_ci["ci_low"],
                    "signed_bias_bca95_high": signed_ci["ci_high"],
                    "mae_bca95_low": mae_ci["ci_low"],
                    "mae_bca95_high": mae_ci["ci_high"],
                    "nonfinite_measurement_count": int(subset["nonfinite_measurement"].sum()),
                    "catastrophic_denominator_count": int(
                        subset["catastrophic_denominator_failure"].sum()
                    ),
                    "near_zero_predicted_edv_count": int(
                        subset["near_zero_predicted_edv"].sum()
                    ),
                    "predicted_esv_gt_edv_count": int(subset["predicted_esv_gt_edv"].sum()),
                    "finite_rvef_outside_0_100_count": int(
                        subset["finite_rvef_outside_0_100"].sum()
                    ),
                }
                if endpoint == "rvef_points":
                    for threshold in (5, 10):
                        valid = subset[f"rvef_error_gt{threshold}"].dropna().astype(bool)
                        interval = wilson_interval(int(valid.sum()), len(valid))
                        row.update(
                            {
                                f"error_gt{threshold}_count": interval["events"],
                                f"error_gt{threshold}_rate": interval["rate"],
                                f"error_gt{threshold}_wilson95_low": interval["ci_low"],
                                f"error_gt{threshold}_wilson95_high": interval["ci_high"],
                            }
                        )
                    for threshold in (40, 45):
                        valid = subset[f"category_disagreement_{threshold}"].dropna().astype(bool)
                        interval = wilson_interval(int(valid.sum()), len(valid))
                        row.update(
                            {
                                f"category_disagreement_{threshold}_count": interval["events"],
                                f"category_disagreement_{threshold}_rate": interval["rate"],
                                f"category_disagreement_{threshold}_wilson95_low": interval["ci_low"],
                                f"category_disagreement_{threshold}_wilson95_high": interval["ci_high"],
                            }
                        )
                    for overlap in (0.90, 0.95):
                        high = subset[subset["rv_dice_minimum_phase"] >= overlap]
                        for threshold in (5, 10):
                            valid = high[f"rvef_error_gt{threshold}"].dropna().astype(bool)
                            interval = wilson_interval(int(valid.sum()), len(valid))
                            key = str(overlap).replace(".", "_")
                            row.update(
                                {
                                    f"silent_gt{threshold}_at_min_dice_ge_{key}_count": interval["events"],
                                    f"silent_gt{threshold}_at_min_dice_ge_{key}_denominator": interval["total"],
                                    f"silent_gt{threshold}_at_min_dice_ge_{key}_rate": interval["rate"],
                                    f"silent_gt{threshold}_at_min_dice_ge_{key}_wilson95_low": interval["ci_low"],
                                    f"silent_gt{threshold}_at_min_dice_ge_{key}_wilson95_high": interval["ci_high"],
                                }
                            )
                rows.append(row)
    return pd.DataFrame(rows)


def _reference_summary(reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in reference.groupby("model", sort=True):
        robust = group["robust_absolute_error_rvef_points"].to_numpy(dtype=float)
        robust_ci = bca_interval(robust, np.mean)
        row: dict[str, Any] = {
            "model": model,
            "patient_count": len(group),
            "reference_spread_rvef_mean": float(group["reference_spread_rvef_points"].mean()),
            "reference_spread_rvef_median": float(group["reference_spread_rvef_points"].median()),
            "robust_rvef_mae": robust_ci["estimate"],
            "robust_rvef_mae_bca95_low": robust_ci["ci_low"],
            "robust_rvef_mae_bca95_high": robust_ci["ci_high"],
        }
        for threshold in (5, 10):
            event = group[f"robust_rvef_error_gt{threshold}"].astype(bool)
            interval = wilson_interval(int(event.sum()), len(event))
            row.update(
                {
                    f"robust_gt{threshold}_count": interval["events"],
                    f"robust_gt{threshold}_rate": interval["rate"],
                    f"robust_gt{threshold}_wilson95_low": interval["ci_low"],
                    f"robust_gt{threshold}_wilson95_high": interval["ci_high"],
                    f"failure_flip_gt{threshold}_count": int(
                        group[f"failure_flip_gt{threshold}"].sum()
                    ),
                }
            )
            for overlap in (0.90, 0.95):
                high = group[group["minimum_phase_rv_dice_both_references"] >= overlap]
                events = high[f"robust_rvef_error_gt{threshold}"].astype(bool)
                interval = wilson_interval(int(events.sum()), len(events))
                key = str(overlap).replace(".", "_")
                row.update(
                    {
                        f"robust_silent_gt{threshold}_at_both_min_dice_ge_{key}_count": interval[
                            "events"
                        ],
                        f"robust_silent_gt{threshold}_at_both_min_dice_ge_{key}_denominator": interval[
                            "total"
                        ],
                        f"robust_silent_gt{threshold}_at_both_min_dice_ge_{key}_rate": interval[
                            "rate"
                        ],
                        f"robust_silent_gt{threshold}_at_both_min_dice_ge_{key}_wilson95_low": interval[
                            "ci_low"
                        ],
                        f"robust_silent_gt{threshold}_at_both_min_dice_ge_{key}_wilson95_high": interval[
                            "ci_high"
                        ],
                    }
                )
        for threshold in (40, 45):
            row[f"reference_category_flip_{threshold}_count"] = int(
                group[f"reference_category_flip_{threshold}"].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _historical_comparison(patients: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = {
        "rv_dice_minimum_phase": "higher",
        "rv_dice_mean_phase": "higher",
        "rv_dice_es": "higher",
        "rv_hd95_mm_maximum_phase": "lower",
        "absolute_error_edv_ml": "lower",
        "absolute_error_esv_ml": "lower",
        "absolute_error_stroke_volume_ml": "lower",
        "absolute_error_rvef_points": "lower",
    }
    for (cohort, policy), group in patients.groupby(["cohort", "reference_policy"], sort=True):
        for stratum, subset in _strata(group):
            for metric, direction in metrics.items():
                pivot = subset.pivot(index="patient_id", columns="model", values=metric).dropna()
                difference = pivot["c2_ensemble"] - pivot["historical_unet"]
                interval = bca_interval(difference, np.mean)
                rows.append(
                    {
                        "cohort": cohort,
                        "reference_policy": policy,
                        "stratum": stratum,
                        "metric": metric,
                        "direction": direction,
                        "difference_definition": "c2_ensemble_minus_historical_unet",
                        "n_paired": len(difference),
                        "mean_difference": interval["estimate"],
                        "bca95_low": interval["ci_low"],
                        "bca95_high": interval["ci_high"],
                    }
                )
    for metric, direction in (
        ("minimum_phase_rv_dice_both_references", "higher"),
        ("robust_absolute_error_rvef_points", "lower"),
    ):
        pivot = reference.pivot(index="patient_id", columns="model", values=metric).dropna()
        difference = pivot["c2_ensemble"] - pivot["historical_unet"]
        interval = bca_interval(difference, np.mean)
        rows.append(
            {
                "cohort": "ACDC_test",
                "reference_policy": "convention_envelope",
                "stratum": "pooled",
                "metric": metric,
                "direction": direction,
                "difference_definition": "c2_ensemble_minus_historical_unet",
                "n_paired": len(difference),
                "mean_difference": interval["estimate"],
                "bca95_low": interval["ci_low"],
                "bca95_high": interval["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def _paired_spearman_interval(x: Iterable[float], y: Iterable[float]) -> dict[str, float]:
    first = np.asarray(list(x), dtype=float)
    second = np.asarray(list(y), dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    estimate = float(stats.spearmanr(first, second).statistic) if len(first) >= 3 else float("nan")
    if len(first) < 5 or np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan")}

    def statistic(a: np.ndarray, b: np.ndarray) -> float:
        return float(stats.spearmanr(a, b).statistic)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.bootstrap(
            (first, second),
            statistic,
            paired=True,
            vectorized=False,
            confidence_level=0.95,
            n_resamples=5000,
            method="BCa",
            random_state=np.random.default_rng(20260812),
        )
    return {
        "estimate": estimate,
        "ci_low": float(result.confidence_interval.low),
        "ci_high": float(result.confidence_interval.high),
    }


def _fold_monitor_outputs(
    patients: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    disagreement = pd.read_csv(FOLD_DISAGREEMENT_PATH, keep_default_na=False)
    acdc_target = reference[reference["model"] == "c2_ensemble"][
        ["cohort", "patient_id", "robust_absolute_error_rvef_points", "robust_rvef_error_gt5"]
    ].rename(
        columns={
            "robust_absolute_error_rvef_points": "target_absolute_rvef_error",
            "robust_rvef_error_gt5": "target_error_gt5",
        }
    )
    mnms_target = patients[
        (patients["cohort"] == "MnMs")
        & (patients["model"] == "c2_ensemble")
        & (patients["reference_policy"] == "mnms")
    ][["cohort", "patient_id", "absolute_error_rvef_points", "rvef_error_gt5"]].rename(
        columns={
            "absolute_error_rvef_points": "target_absolute_rvef_error",
            "rvef_error_gt5": "target_error_gt5",
        }
    )
    targets = pd.concat([acdc_target, mnms_target], ignore_index=True)
    merged = disagreement.merge(targets, on=["cohort", "patient_id"], how="left", validate="one_to_one")
    if merged["target_absolute_rvef_error"].isna().any():
        raise RuntimeError("Fold disagreement target merge is incomplete")

    risk_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    for cohort, cohort_group in merged.groupby("cohort", sort=True):
        stratum_groups = [("pooled", cohort_group)]
        if cohort == "MnMs":
            stratum_groups.extend(
                (f"vendor_{vendor}", group) for vendor, group in cohort_group.groupby("vendor", sort=True)
            )
        for stratum, group in stratum_groups:
            ordered = group.sort_values(
                ["fold_rvef_range_points", "patient_id"], ascending=[True, True]
            ).reset_index(drop=True)
            for coverage in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00):
                keep = max(1, int(math.ceil(len(ordered) * coverage)))
                retained = ordered.iloc[:keep]
                risk_ci = bca_interval(retained["target_absolute_rvef_error"], np.mean)
                event = retained["target_error_gt5"].astype(bool)
                event_ci = wilson_interval(int(event.sum()), len(event))
                risk_rows.append(
                    {
                        "cohort": cohort,
                        "stratum": stratum,
                        "coverage": coverage,
                        "retained_patients": keep,
                        "total_patients": len(ordered),
                        "mean_absolute_rvef_error": risk_ci["estimate"],
                        "mean_absolute_rvef_error_bca95_low": risk_ci["ci_low"],
                        "mean_absolute_rvef_error_bca95_high": risk_ci["ci_high"],
                        "error_gt5_rate": event_ci["rate"],
                        "error_gt5_wilson95_low": event_ci["ci_low"],
                        "error_gt5_wilson95_high": event_ci["ci_high"],
                    }
                )
            discrimination = discrimination_metrics(
                group["target_error_gt5"], group["fold_rvef_range_points"]
            )
            correlation = _paired_spearman_interval(
                group["target_absolute_rvef_error"], group["fold_rvef_range_points"]
            )
            risk_here = [row for row in risk_rows if row["cohort"] == cohort and row["stratum"] == stratum]
            full = next(row for row in risk_here if row["coverage"] == 1.0)[
                "mean_absolute_rvef_error"
            ]
            at80 = next(row for row in risk_here if row["coverage"] == 0.8)[
                "mean_absolute_rvef_error"
            ]
            improvement = float((full - at80) / full) if np.isfinite(full) and full > 0 else float("nan")
            monitor_rows.append(
                {
                    "cohort": cohort,
                    "stratum": stratum,
                    "signal": "fold_rvef_range_points",
                    **discrimination,
                    "spearman_absolute_error": correlation["estimate"],
                    "spearman_bca95_low": correlation["ci_low"],
                    "spearman_bca95_high": correlation["ci_high"],
                    "relative_risk_improvement_at_80pct_coverage": improvement,
                }
            )
    return merged, pd.DataFrame(monitor_rows), pd.DataFrame(risk_rows)


def _decision(
    reference: pd.DataFrame,
    reference_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    monitor: pd.DataFrame,
) -> dict[str, Any]:
    c2 = reference[reference["model"] == "c2_ensemble"].copy()
    summary = reference_summary.set_index("model").loc["c2_ensemble"]
    high = c2[c2["minimum_phase_rv_dice_both_references"] >= 0.90]
    silent_gt5 = int(high["robust_rvef_error_gt5"].sum())
    silent_gt10 = int(high["robust_rvef_error_gt10"].sum())
    robust_gt10 = int(c2["robust_rvef_error_gt10"].sum())
    material = bool(
        float(summary["robust_rvef_mae"]) >= 4.0
        or float(summary["robust_gt5_rate"]) >= 0.10
        or robust_gt10 >= 2
    )
    leave_one = high.sort_values("robust_absolute_error_rvef_points", ascending=False).iloc[1:]
    leave_one_persists = bool(leave_one["robust_rvef_error_gt5"].any())
    monitor_row = monitor[
        (monitor["cohort"] == "ACDC_test") & (monitor["stratum"] == "pooled")
    ].iloc[0]
    monitor_weak = bool(
        int(monitor_row["positives"]) < 10
        or int(monitor_row["negatives"]) < 10
        or (np.isfinite(monitor_row["auroc"]) and float(monitor_row["auroc"]) < 0.70)
        or (
            np.isfinite(monitor_row["auroc_ci_low"])
            and float(monitor_row["auroc_ci_low"]) <= 0.50 <= float(monitor_row["auroc_ci_high"])
        )
        or (
            np.isfinite(monitor_row["relative_risk_improvement_at_80pct_coverage"])
            and float(monitor_row["relative_risk_improvement_at_80pct_coverage"]) < 0.20
        )
    )
    strong_checks = {
        "GO_C2_DEVELOPMENT_QUALIFIED": True,
        "ACDC_clinically_material_robust_error": material,
        "at_least_two_robust_gt5_silent_failures_at_both_reference_min_dice_ge_0_90": silent_gt5
        >= 2,
        "original_refined_sensitivity_addressed_by_convention_envelope": True,
        "invalid_and_pathological_cases_reported": True,
        "fold_disagreement_monitor_weak_or_incomplete": monitor_weak,
        "leave_one_worst_patient_out_persistence": leave_one_persists,
    }
    paired_dice = comparison[
        (comparison["reference_policy"] == "convention_envelope")
        & (comparison["metric"] == "minimum_phase_rv_dice_both_references")
    ].iloc[0]
    historical_mae = float(
        reference_summary.set_index("model").loc["historical_unet", "robust_rvef_mae"]
    )
    reduction = (
        float((historical_mae - summary["robust_rvef_mae"]) / historical_mae)
        if historical_mae > 0
        else float("nan")
    )
    weak_checks = {
        "paired_C2_minus_historical_minimum_phase_Dice_BCa_lower_above_0": float(
            paired_dice["bca95_low"]
        )
        > 0,
        "C2_robust_RVEF_MAE_at_least_30_percent_below_historical": reduction >= 0.30,
        "no_more_than_one_C2_robust_gt5_silent_failure": silent_gt5 <= 1,
        "zero_C2_robust_gt10_silent_failures": silent_gt10 == 0,
    }
    if all(strong_checks.values()):
        decision = "HIGH_DICE_DOES_NOT_CERTIFY_MEASUREMENT_RELIABILITY"
    elif all(weak_checks.values()):
        decision = "MEASUREMENT_FAILURE_LARGELY_EXPLAINED_BY_WEAK_SEGMENTATION"
    else:
        decision = "RELIABILITY_RESULT_INCONCLUSIVE"
    return {
        "schema_version": 1,
        "decision": decision,
        "generated_at_utc": common.now_utc(),
        "baseline_qualification": "GO_C2_DEVELOPMENT_QUALIFIED",
        "strong_result_checks": strong_checks,
        "weak_segmentation_explanation_checks": weak_checks,
        "factual_counts": {
            "ACDC_patient_count": len(c2),
            "high_overlap_both_references_patient_count": len(high),
            "robust_gt5_silent_failure_count": silent_gt5,
            "robust_gt10_silent_failure_count": silent_gt10,
            "robust_gt10_all_patient_count": robust_gt10,
            "C2_robust_rvef_mae": float(summary["robust_rvef_mae"]),
            "historical_robust_rvef_mae": historical_mae,
            "relative_robust_mae_reduction_vs_historical": reduction,
        },
        "interpretation_constraints": [
            "The two-reference range is a convention envelope, not a confidence interval.",
            "Neither ACDC annotation convention is claimed anatomically correct.",
            "M&Ms vendor results are descriptive and do not identify annotation policy.",
            "No training, calibration, post-processing, or threshold selection occurred.",
        ],
    }


def _plot_results(
    patients: pd.DataFrame,
    reference: pd.DataFrame,
    risk: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    c2_acdc = reference[reference["model"] == "c2_ensemble"]
    c2_mnms = patients[
        (patients["cohort"] == "MnMs")
        & (patients["model"] == "c2_ensemble")
        & (patients["reference_policy"] == "mnms")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].scatter(
        c2_acdc["minimum_phase_rv_dice_both_references"],
        c2_acdc["robust_absolute_error_rvef_points"],
        s=24,
        color="#2364AA",
        alpha=0.8,
    )
    axes[0].axvline(0.90, color="#B23A48", linestyle="--", linewidth=1)
    axes[0].axhline(5, color="#B23A48", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Minimum RV Dice across phases and references", ylabel="Robust RVEF error (points)", title="ACDC convention envelope")
    axes[1].scatter(
        c2_mnms["rv_dice_minimum_phase"],
        c2_mnms["absolute_error_rvef_points"],
        s=14,
        color="#3A7D44",
        alpha=0.55,
    )
    axes[1].axvline(0.90, color="#B23A48", linestyle="--", linewidth=1)
    axes[1].axhline(5, color="#B23A48", linestyle=":", linewidth=1)
    axes[1].set(xlabel="Minimum-phase RV Dice", ylabel="Absolute RVEF error (points)", title="M&Ms pooled")
    fig.savefig(FIGURES["overlap"], dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    ax.scatter(
        c2_acdc["original_reference_rvef_points"],
        c2_acdc["refined_reference_rvef_points"],
        c=c2_acdc["robust_absolute_error_rvef_points"],
        cmap="viridis",
        s=34,
    )
    limits = [
        min(c2_acdc["original_reference_rvef_points"].min(), c2_acdc["refined_reference_rvef_points"].min()),
        max(c2_acdc["original_reference_rvef_points"].max(), c2_acdc["refined_reference_rvef_points"].max()),
    ]
    ax.plot(limits, limits, color="#555555", linewidth=1)
    ax.set(xlabel="Original-reference RVEF", ylabel="Refined-reference RVEF", title="ACDC reference-policy shift")
    fig.savefig(FIGURES["reference"], dpi=240)
    plt.close(fig)

    pivot = reference.pivot(index="patient_id", columns="model", values="robust_absolute_error_rvef_points")
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row["historical_unet"], row["c2_ensemble"]], color="#AAAAAA", alpha=0.3)
    ax.scatter(np.zeros(len(pivot)), pivot["historical_unet"], color="#9C3D54", s=20, label="Historical U-Net")
    ax.scatter(np.ones(len(pivot)), pivot["c2_ensemble"], color="#2364AA", s=20, label="C2 ensemble")
    ax.set_xticks([0, 1], ["Historical U-Net", "C2 ensemble"])
    ax.set_ylabel("Robust RVEF error (points)")
    ax.set_title("Paired ACDC comparison")
    fig.savefig(FIGURES["historical"], dpi=240)
    plt.close(fig)

    vendors = sorted(c2_mnms["vendor"].unique())
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    arrays = [c2_mnms.loc[c2_mnms["vendor"] == vendor, "absolute_error_rvef_points"].dropna() for vendor in vendors]
    ax.boxplot(arrays, labels=vendors, showfliers=False)
    ax.set(xlabel="Vendor", ylabel="Absolute RVEF error (points)", title="M&Ms vendor-stratified descriptive results")
    fig.savefig(FIGURES["vendor"], dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    for (cohort, stratum), group in risk.groupby(["cohort", "stratum"], sort=True):
        if stratum != "pooled":
            continue
        ax.plot(group["coverage"], group["mean_absolute_rvef_error"], marker="o", label=cohort)
    ax.set(xlabel="Coverage retained", ylabel="Mean absolute RVEF error (points)", title="Fold-disagreement risk coverage")
    ax.legend(frameon=False)
    fig.savefig(FIGURES["risk"], dpi=240)
    plt.close(fig)


def _write_report(
    decision: dict[str, Any], reference_summary: pd.DataFrame, monitor: pd.DataFrame
) -> None:
    c2 = reference_summary.set_index("model").loc["c2_ensemble"]
    hist = reference_summary.set_index("model").loc["historical_unet"]
    acdc_monitor = monitor[
        (monitor["cohort"] == "ACDC_test") & (monitor["stratum"] == "pooled")
    ].iloc[0]
    facts = decision["factual_counts"]
    text = f"""# C2 Reliability Evaluation

## Frozen Instrument

The accepted baseline decision is `GO_C2_DEVELOPMENT_QUALIFIED`.
Input validation checks the expected NIfTI float32 affine serialization;
the bounded header correction does not resample or interpolate voxels. The
measurement instrument was the official nnU-Net v2.8.1 2D five-fold ensemble,
using `checkpoint_final` for every fold. Each fold used the qualified OOF
preprocessing and native-export contract. The primary segmentation was made by
nnU-Net's official probability ensembling implementation. No training,
calibration, post-processing, fold selection, or threshold change occurred.

## Factual Decision

**`{decision['decision']}`**

- ACDC patients evaluated: {facts['ACDC_patient_count']}.
- C2 convention-envelope robust RVEF MAE: {facts['C2_robust_rvef_mae']:.3f} points.
- Historical U-Net convention-envelope robust RVEF MAE: {facts['historical_robust_rvef_mae']:.3f} points.
- Patients high-overlap against both references: {facts['high_overlap_both_references_patient_count']}.
- Robust >5-point silent failures: {facts['robust_gt5_silent_failure_count']}.
- Robust >10-point silent failures: {facts['robust_gt10_silent_failure_count']}.
- C2 robust >5 rate: {float(c2['robust_gt5_rate']):.3f} ({int(c2['robust_gt5_count'])}/50).
- Historical robust >5 rate: {float(hist['robust_gt5_rate']):.3f} ({int(hist['robust_gt5_count'])}/50).
- ACDC fold-range AUROC for robust >5 error: {float(acdc_monitor['auroc']):.3f}.

The exact predeclared decision checks are in `reliability_decision.json`; this
report does not replace them with a post-hoc interpretation.

## Interpretation Boundaries

The range between original and refined ACDC measurements is called a
**convention envelope**, not a confidence interval. The analysis does not claim
that either annotation convention is anatomically correct. M&Ms vendor results
are descriptive and do not identify vendor-specific annotation policy.

M&Ms-2 is outside this analysis.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def run_analysis() -> dict[str, Any]:
    common.load_protocol()
    if not INFERENCE_MANIFEST_PATH.is_file():
        raise RuntimeError("Frozen reliability inference is incomplete")
    if ANALYSIS_MANIFEST_PATH.exists():
        raise RuntimeError("Analysis artifacts already exist; replacement is forbidden")
    common.write_status("analysis_running", started_at_utc=common.now_utc())
    cases = load_case_manifest()
    case_metrics = _case_metrics(cases)
    case_metrics.to_csv(CASE_OUTPUT, index=False)
    patients = _patient_metrics(cases, case_metrics)
    patients.to_csv(PATIENT_OUTPUT, index=False)
    reference = _reference_policy_table(patients)
    reference.to_csv(REFERENCE_POLICY, index=False)
    segmentation = _segmentation_battery(patients)
    segmentation.to_csv(SEGMENTATION_BATTERY, index=False)
    measurement = _measurement_battery(patients)
    measurement.to_csv(MEASUREMENT_BATTERY, index=False)
    reference_summary = _reference_summary(reference)
    reference_summary.to_csv(REFERENCE_SUMMARY, index=False)
    comparison = _historical_comparison(patients, reference)
    comparison.to_csv(HISTORICAL_COMPARISON, index=False)
    fold_patient, monitor, risk = _fold_monitor_outputs(patients, reference)
    fold_patient.to_csv(common.OUTPUT_DIR / "fold_disagreement.csv", index=False)
    monitor.to_csv(FOLD_MONITOR, index=False)
    risk.to_csv(RISK_COVERAGE, index=False)
    decision = _decision(reference, reference_summary, comparison, monitor)
    common.write_json(DECISION_PATH, decision)
    _plot_results(patients, reference, risk)
    _write_report(decision, reference_summary, monitor)
    output_paths = [
        CASE_OUTPUT,
        PATIENT_OUTPUT,
        SEGMENTATION_BATTERY,
        MEASUREMENT_BATTERY,
        REFERENCE_POLICY,
        REFERENCE_SUMMARY,
        HISTORICAL_COMPARISON,
        common.OUTPUT_DIR / "fold_disagreement.csv",
        FOLD_MONITOR,
        RISK_COVERAGE,
        DECISION_PATH,
        REPORT_PATH,
        *FIGURES.values(),
    ]
    manifest = {
        "schema_version": 1,
        "decision": "RELIABILITY_ANALYSIS_COMPLETE",
        "factual_reliability_decision": decision["decision"],
        "generated_at_utc": common.now_utc(),
        "statistical_unit": "patient",
        "bootstrap": {"method": "BCa", "resamples": 5000, "seed": 20260812},
        "proportion_intervals": "Wilson",
        "outputs": common.records_for_paths(output_paths),
        "training": False,
        "no_MnMs2_content_access": True,
    }
    common.write_json(ANALYSIS_MANIFEST_PATH, manifest)
    common.write_status(
        "analysis_completed",
        reliability_decision=decision["decision"],
        analysis_manifest=common.file_record(ANALYSIS_MANIFEST_PATH),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "infer", "analyze", "all"), required=True)
    parser.add_argument("--acdc-dir", type=Path, required=True)
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--mnms-dir", type=Path, required=True)
    parser.add_argument("--mnms-metadata", type=Path, required=True)
    parser.add_argument("--nnunet-root", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    common.configure(args)
    import importlib
    import reliability_data
    import reliability_inference
    importlib.reload(reliability_data)
    importlib.reload(reliability_inference)
    global C2_ROOT, HISTORICAL_ROOT, FOLD_DISAGREEMENT_PATH, INFERENCE_MANIFEST_PATH
    C2_ROOT = reliability_inference.C2_ROOT
    HISTORICAL_ROOT = reliability_inference.HISTORICAL_ROOT
    FOLD_DISAGREEMENT_PATH = reliability_inference.FOLD_DISAGREEMENT_PATH
    INFERENCE_MANIFEST_PATH = reliability_inference.INFERENCE_MANIFEST_PATH
    for name, filename in {
        "CASE_OUTPUT": "reliability_per_case.csv", "PATIENT_OUTPUT": "reliability_per_patient.csv",
        "SEGMENTATION_BATTERY": "segmentation_battery.csv", "MEASUREMENT_BATTERY": "measurement_battery.csv",
        "REFERENCE_POLICY": "reference_policy_analysis.csv", "REFERENCE_SUMMARY": "reference_policy_summary.csv",
        "HISTORICAL_COMPARISON": "historical_paired_comparison.csv", "FOLD_MONITOR": "fold_monitor_summary.csv",
        "RISK_COVERAGE": "risk_coverage.csv", "DECISION_PATH": "reliability_decision.json",
        "REPORT_PATH": "RELIABILITY_REPORT.md", "ANALYSIS_MANIFEST_PATH": "reliability_analysis_manifest.json",
    }.items():
        globals()[name] = common.OUTPUT_DIR / filename
    for name, path in FIGURES.items():
        FIGURES[name] = common.OUTPUT_DIR / path.name
    if args.stage in ("prepare", "all"):
        reliability_data.prepare_restricted_inputs()
    if args.stage in ("infer", "all"):
        reliability_inference.run_all_inference()
    if args.stage in ("analyze", "all"):
        print(run_analysis()["factual_reliability_decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
