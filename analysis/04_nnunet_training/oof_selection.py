"""Global segmentation-only checkpoint-rule selection and bootstrap utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

RULES = ("checkpoint_final", "checkpoint_best")
PRIMARY_MARGIN = 0.002
SELECTION_METRICS = (
    ("mean_rv_dice", "rv_dice_mean_phase", "higher"),
    ("minimum_phase_rv_dice", "rv_dice_minimum_phase", "higher"),
    ("rv_dice_es", "rv_dice_es", "higher"),
    ("rv_hd95_mm", "rv_hd95_mm_mean_phase", "lower"),
    ("rv_normalized_surface_dice_2mm", "rv_nsd_2mm_mean_phase", "higher"),
)


def bca_interval(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 5000,
    seed: int = 20260804,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("BCa input must be a finite one-dimensional patient array")
    estimate = float(statistic(array))
    if np.all(array == array[0]):
        return estimate, estimate
    result = bootstrap(
        (array,),
        statistic,
        vectorized=False,
        paired=False,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def summarize_rule(frame: pd.DataFrame) -> dict[str, Any]:
    rules = frame["checkpoint_rule"].unique().tolist()
    if len(rules) != 1:
        raise RuntimeError("Rule summary requires exactly one checkpoint rule")
    if len(frame) != 100 or frame["patient"].nunique() != 100:
        raise RuntimeError("Global rule summary requires 100 development patients")
    columns = {column for _, column, _ in SELECTION_METRICS}
    columns.update(
        {
            "lv_dice_mean_phase",
            "myocardium_dice_mean_phase",
            "rv_dice_ed",
            "lv_dice_minimum_phase",
            "myocardium_dice_minimum_phase",
        }
    )
    if not np.isfinite(frame[sorted(columns)].to_numpy(dtype=float)).all():
        raise FloatingPointError(f"Nonfinite OOF metrics for {rules[0]}")
    return {
        "checkpoint_rule": rules[0],
        "patient_count": len(frame),
        "valid_patient_count": int(frame["valid_patient"].sum()),
        "invalid_patient_rate": float(1.0 - frame["valid_patient"].mean()),
        "mean_rv_dice": float(frame["rv_dice_mean_phase"].mean()),
        "minimum_phase_rv_dice": float(frame["rv_dice_minimum_phase"].mean()),
        "rv_dice_ed": float(frame["rv_dice_ed"].mean()),
        "rv_dice_es": float(frame["rv_dice_es"].mean()),
        "rv_hd95_mm": float(frame["rv_hd95_mm_mean_phase"].mean()),
        "rv_normalized_surface_dice_2mm": float(frame["rv_nsd_2mm_mean_phase"].mean()),
        "mean_lv_dice": float(frame["lv_dice_mean_phase"].mean()),
        "minimum_phase_lv_dice": float(frame["lv_dice_minimum_phase"].mean()),
        "mean_myocardium_dice": float(frame["myocardium_dice_mean_phase"].mean()),
        "minimum_phase_myocardium_dice": float(
            frame["myocardium_dice_minimum_phase"].mean()
        ),
    }


def summarize_global_rules(patient_metrics: pd.DataFrame) -> pd.DataFrame:
    if set(patient_metrics["checkpoint_rule"]) != set(RULES):
        raise RuntimeError("OOF metrics must contain exactly final and best rules")
    summaries = [
        summarize_rule(patient_metrics[patient_metrics["checkpoint_rule"] == rule])
        for rule in RULES
    ]
    return pd.DataFrame(summaries)


def select_global_rule(summary: pd.DataFrame) -> dict[str, Any]:
    if set(summary["checkpoint_rule"]) != set(RULES) or len(summary) != 2:
        raise RuntimeError("Selection requires exactly two global rule summaries")
    indexed = summary.set_index("checkpoint_rule")
    trace: list[dict[str, Any]] = []
    primary_name, _, _ = SELECTION_METRICS[0]
    final_primary = float(indexed.loc["checkpoint_final", primary_name])
    best_primary = float(indexed.loc["checkpoint_best", primary_name])
    primary_delta = final_primary - best_primary
    trace.append(
        {
            "criterion": primary_name,
            "checkpoint_final": final_primary,
            "checkpoint_best": best_primary,
            "final_minus_best": primary_delta,
            "direction": "higher",
            "equivalence_margin": PRIMARY_MARGIN,
            "decisive": abs(primary_delta) > PRIMARY_MARGIN,
        }
    )
    if abs(primary_delta) > PRIMARY_MARGIN:
        selected = "checkpoint_final" if primary_delta > 0 else "checkpoint_best"
        return {
            "selected_global_rule": selected,
            "selection_basis": "mean_rv_dice_outside_primary_equivalence_margin",
            "primary_equivalence_margin": PRIMARY_MARGIN,
            "selection_trace": trace,
            "same_rule_all_folds": True,
        }
    for metric_name, _, direction in SELECTION_METRICS[1:]:
        final_value = float(indexed.loc["checkpoint_final", metric_name])
        best_value = float(indexed.loc["checkpoint_best", metric_name])
        delta = final_value - best_value
        decisive = delta != 0.0
        trace.append(
            {
                "criterion": metric_name,
                "checkpoint_final": final_value,
                "checkpoint_best": best_value,
                "final_minus_best": delta,
                "direction": direction,
                "equivalence_margin": None,
                "decisive": decisive,
            }
        )
        if decisive:
            if direction == "higher":
                selected = "checkpoint_final" if delta > 0 else "checkpoint_best"
            else:
                selected = "checkpoint_final" if delta < 0 else "checkpoint_best"
            return {
                "selected_global_rule": selected,
                "selection_basis": metric_name,
                "primary_equivalence_margin": PRIMARY_MARGIN,
                "selection_trace": trace,
                "same_rule_all_folds": True,
            }
    return {
        "selected_global_rule": "checkpoint_final",
        "selection_basis": "exact_tie_fallback",
        "primary_equivalence_margin": PRIMARY_MARGIN,
        "selection_trace": trace,
        "same_rule_all_folds": True,
    }


def paired_rule_comparisons(
    patient_metrics: pd.DataFrame,
    *,
    n_resamples: int = 5000,
    seed: int = 20260804,
) -> pd.DataFrame:
    final = (
        patient_metrics[patient_metrics["checkpoint_rule"] == "checkpoint_final"]
        .set_index("patient")
        .sort_index()
    )
    best = (
        patient_metrics[patient_metrics["checkpoint_rule"] == "checkpoint_best"]
        .set_index("patient")
        .sort_index()
    )
    if list(final.index) != list(best.index) or len(final) != 100:
        raise RuntimeError("Global rule comparison is not paired by all 100 patients")
    rows = []
    for offset, (metric_name, column, direction) in enumerate(SELECTION_METRICS):
        delta = final[column].to_numpy(dtype=float) - best[column].to_numpy(dtype=float)
        low, high = bca_interval(
            delta,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        rows.append(
            {
                "metric": metric_name,
                "patient_count": len(delta),
                "checkpoint_final_mean": float(final[column].mean()),
                "checkpoint_best_mean": float(best[column].mean()),
                "paired_final_minus_best": float(delta.mean()),
                "paired_delta_bca95_low": low,
                "paired_delta_bca95_high": high,
                "direction": direction,
                "bootstrap_unit": "patient",
                "bootstrap_method": "BCa",
                "bootstrap_resamples": n_resamples,
            }
        )
    return pd.DataFrame(rows)
