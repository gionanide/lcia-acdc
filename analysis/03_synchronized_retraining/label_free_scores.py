#!/usr/bin/env python
"""Phase 5: deployment monitor reassessment after synchronized convention training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C2
import convention_utils as U


THRESHOLDS = (40.0, 45.0)


def normalize_endpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ("original_reference", "refined_reference", "original_model", "refined_model"):
        if f"{prefix}_EDV" not in out and f"{prefix}_ED_ml" in out:
            out[f"{prefix}_EDV"] = out[f"{prefix}_ED_ml"]
        if f"{prefix}_ESV" not in out and f"{prefix}_ES_ml" in out:
            out[f"{prefix}_ESV"] = out[f"{prefix}_ES_ml"]
        lower = f"{prefix}_rvef"
        if f"{prefix}_RVEF" not in out and lower in out:
            out[f"{prefix}_RVEF"] = out[lower]
    return out


def interval_distance(value: float, a: float, b: float) -> float:
    return U.interval_distance(float(value), float(a), float(b))


def plausibility_score(row: pd.Series) -> float:
    score = 0.0
    for arm in ("original_model", "refined_model"):
        edv = float(row[f"{arm}_EDV"])
        esv = float(row[f"{arm}_ESV"])
        score += max(0.0, esv - edv)
        score += max(0.0, 20.0 - edv)
        score += max(0.0, 10.0 - esv)
        score += max(0.0, edv - 300.0)
        score += max(0.0, esv - 250.0)
    return float(score)


def build_scores(
    endpoint: pd.DataFrame,
    spatial: pd.DataFrame,
    calibration: pd.DataFrame,
    novelty: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    endpoint = normalize_endpoint_columns(endpoint)
    rows = []
    spatial_agg = (
        spatial.groupby(["seed", "pid"], as_index=False)
        .agg(
            support_boundary_disagreement=("rv_presence_disagreement_slices", "sum"),
            support_length_abs_shift=("support_length_shift", lambda x: float(np.sum(np.abs(x)))),
            model_directional_change_voxels=("model_change_voxels", "sum"),
            reference_directional_change_voxels=("reference_change_voxels", "sum"),
            soft_probability_shift_mass=(
                "soft_mass_inside_reference_change",
                lambda x: float(np.sum(np.abs(np.asarray(x, dtype=float)))),
            ),
            mean_directional_precision=("precision", "mean"),
            mean_directional_recall=("recall", "mean"),
        )
    )
    cal_rvef = calibration[
        (calibration["endpoint"] == "RVEF")
        & (calibration["method"] == "volume_affine")
        & (calibration["label_budget"] == calibration["label_budget"].max())
        & (calibration["direction"] == "original_to_refined")
    ][["seed", "pid", "abs_error"]].rename(columns={"abs_error": "calibration_residual_rvef"})
    arch = novelty[["pid", "range_pred_rvef", "sd_pred_rvef", "mean_pred_rvef"]].rename(
        columns={
            "range_pred_rvef": "ordinary_architecture_rvef_range",
            "sd_pred_rvef": "ordinary_architecture_rvef_sd",
            "mean_pred_rvef": "ordinary_architecture_mean_rvef",
        }
    )
    seed_disagreement = (
        endpoint.groupby("pid", as_index=False)
        .agg(
            same_convention_seed_range=("original_model_RVEF", lambda x: float(np.max(x) - np.min(x))),
            same_convention_seed_sd=("original_model_RVEF", "std"),
        )
    )
    merged = endpoint.merge(spatial_agg, on=["seed", "pid"], how="left")
    merged = merged.merge(cal_rvef, on=["seed", "pid"], how="left")
    merged = merged.merge(arch, on="pid", how="left")
    merged = merged.merge(seed_disagreement, on="pid", how="left")
    rng = np.random.default_rng(seed)
    random_scores = {
        (int(row.seed), str(row.pid)): float(rng.random())
        for row in merged[["seed", "pid"]].itertuples(index=False)
    }
    for row in merged.itertuples(index=False):
        record = row._asdict()
        original_ref = float(record["original_reference_RVEF"])
        refined_ref = float(record["refined_reference_RVEF"])
        original_pred = float(record["original_model_RVEF"])
        refined_pred = float(record["refined_model_RVEF"])
        midpoint_pred = 0.5 * (original_pred + refined_pred)
        reference_spread = abs(refined_ref - original_ref)
        failure_original = abs(midpoint_pred - original_ref) > 5.0
        failure_refined = abs(midpoint_pred - refined_ref) > 5.0
        entropy_cols = [
            c
            for c in (
                "original_model_entropy_ED",
                "original_model_entropy_ES",
                "refined_model_entropy_ED",
                "refined_model_entropy_ES",
            )
            if c in record
        ]
        entropy = float(np.nanmean([record[c] for c in entropy_cols])) if entropy_cols else np.nan
        out = {
            "seed": int(record["seed"]),
            "pid": str(record["pid"]),
            "original_reference_RVEF": original_ref,
            "refined_reference_RVEF": refined_ref,
            "reference_shift": refined_ref - original_ref,
            "reference_spread": reference_spread,
            "original_model_RVEF": original_pred,
            "refined_model_RVEF": refined_pred,
            "midpoint_model_RVEF": midpoint_pred,
            "model_shift": refined_pred - original_pred,
            "convention_disagreement": abs(refined_pred - original_pred),
            "support_boundary_disagreement": float(record.get("support_boundary_disagreement", np.nan)),
            "support_length_abs_shift": float(record.get("support_length_abs_shift", np.nan)),
            "model_directional_change_voxels": float(record.get("model_directional_change_voxels", np.nan)),
            "reference_directional_change_voxels": float(record.get("reference_directional_change_voxels", np.nan)),
            "soft_probability_shift_mass": float(record.get("soft_probability_shift_mass", np.nan)),
            "calibration_residual_rvef": float(record.get("calibration_residual_rvef", np.nan)),
            "ordinary_architecture_rvef_range": float(record.get("ordinary_architecture_rvef_range", np.nan)),
            "ordinary_architecture_rvef_sd": float(record.get("ordinary_architecture_rvef_sd", np.nan)),
            "same_convention_seed_range": float(record.get("same_convention_seed_range", np.nan)),
            "same_convention_seed_sd": float(record.get("same_convention_seed_sd", np.nan)),
            "entropy": entropy,
            "edv_esv_plausibility_score": plausibility_score(pd.Series(record)),
            "random_uniform_rank": random_scores[(int(record["seed"]), str(record["pid"]))],
            "reference_spread_gt5": int(reference_spread > 5.0),
            "reference_failure_label_flip_5": int(failure_original != failure_refined),
            "robust_failure_outside_envelope_gt5": int(interval_distance(midpoint_pred, original_ref, refined_ref) > 5.0),
        }
        for threshold in THRESHOLDS:
            out[f"reference_threshold_instability_{int(threshold)}"] = int(
                (original_ref < threshold) != (refined_ref < threshold)
            )
            out[f"model_threshold_instability_{int(threshold)}"] = int(
                (original_pred < threshold) != (refined_pred < threshold)
            )
        rows.append(out)
    return pd.DataFrame(rows)


def score_columns() -> List[str]:
    return [
        "convention_disagreement",
        "support_boundary_disagreement",
        "support_length_abs_shift",
        "model_directional_change_voxels",
        "soft_probability_shift_mass",
        "calibration_residual_rvef",
        "ordinary_architecture_rvef_range",
        "ordinary_architecture_rvef_sd",
        "same_convention_seed_range",
        "entropy",
        "edv_esv_plausibility_score",
        "random_uniform_rank",
    ]


def target_columns() -> List[str]:
    return [
        "reference_spread_gt5",
        "reference_failure_label_flip_5",
        "reference_threshold_instability_40",
        "reference_threshold_instability_45",
        "robust_failure_outside_envelope_gt5",
    ]


def metric_ci(
    frame: pd.DataFrame,
    target: str,
    score: str,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    point = U.binary_metrics(frame[target], frame[score])
    pids = np.asarray(sorted(frame["pid"].unique()))
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {"auroc": [], "auprc": [], "spearman": []}
    for _ in range(n_boot):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        sample = pd.concat([frame[frame["pid"] == pid] for pid in sampled], ignore_index=True)
        vals = U.binary_metrics(sample[target], sample[score])
        for key in draws:
            draws[key].append(vals[key])
    out = dict(point)
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[f"{key}_ci_low"] = float(np.percentile(arr, 2.5)) if arr.size else np.nan
        out[f"{key}_ci_high"] = float(np.percentile(arr, 97.5)) if arr.size else np.nan
    return out


def build_metrics(scores: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for target in target_columns():
        for score in score_columns():
            vals = metric_ci(scores, target, score, n_boot, U.stable_int(seed, target, score) % (2**32))
            rows.append({"target": target, "score": score, **vals})
    return pd.DataFrame(rows)


def risk_coverage_one(frame: pd.DataFrame, target: str, score: str, coverages: Sequence[float]) -> pd.DataFrame:
    valid = frame[["pid", target, score]].dropna()
    valid = valid.sort_values(score, ascending=True, kind="mergesort")
    rows = []
    n = len(valid)
    if n == 0:
        return pd.DataFrame()
    for coverage in coverages:
        k = max(1, int(np.ceil(n * float(coverage))))
        retained = valid.iloc[:k]
        excluded = valid.iloc[k:]
        rows.append(
            {
                "target": target,
                "score": score,
                "coverage": float(coverage),
                "n_retained": int(len(retained)),
                "retained_event_rate": float(retained[target].mean()),
                "excluded_event_rate": float(excluded[target].mean()) if len(excluded) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_risk_coverage(scores: pd.DataFrame) -> pd.DataFrame:
    coverages = np.round(np.linspace(0.5, 1.0, 11), 2)
    frames = []
    for target in target_columns():
        for score in score_columns():
            frames.append(risk_coverage_one(scores, target, score, coverages))
    return pd.concat(frames, ignore_index=True)


def make_figure(risk: pd.DataFrame, out_dir: Path) -> None:
    target = "reference_threshold_instability_45"
    selected = [
        "convention_disagreement",
        "ordinary_architecture_rvef_range",
        "same_convention_seed_range",
        "entropy",
        "random_uniform_rank",
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    for score in selected:
        g = risk[(risk["target"] == target) & (risk["score"] == score)]
        if len(g):
            ax.plot(g["coverage"], g["retained_event_rate"], marker="o", label=score)
    ax.set_xlabel("Coverage retained after excluding highest-score patients")
    ax.set_ylabel("Retained threshold-instability rate")
    ax.set_title("Convention Monitor Risk-Coverage")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_monitor_v2_risk_coverage.png")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, scores: pd.DataFrame, out_dir: Path) -> None:
    best = (
        metrics.sort_values(["target", "auroc"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()[["target", "score", "auroc", "auprc", "spearman", "n_events", "event_rate"]]
    )
    conv = metrics[metrics["score"] == "convention_disagreement"][
        ["target", "auroc", "auprc", "spearman", "n_events", "event_rate"]
    ].copy()
    lines = [
        "# Monitor V2 Reassessment",
        "",
        "This phase evaluates ground-truth-free deployment signals against convention-envelope targets on ACDC patients 101-150. The two-reference range is called a convention envelope, not a confidence interval.",
        "",
        "## Convention disagreement metrics",
        "",
        conv.to_string(index=False),
        "",
        "## Best score per target",
        "",
        best.to_string(index=False),
        "",
        "## Score definitions",
        "",
        "- `convention_disagreement`: absolute RVEF difference between refined-trained and original-trained synchronized U-Nets.",
        "- `ordinary_architecture_rvef_range`: existing four-architecture RVEF range from the Stage A novelty screen.",
        "- `same_convention_seed_range`: original-trained U-Net RVEF range across synchronized training seeds.",
        "- `entropy`: mean softmax entropy across ED/ES and both convention-trained U-Nets.",
        "- `edv_esv_plausibility_score`: penalty for impossible or extreme predicted EDV/ESV values.",
        "",
        "No result here claims that either annotation convention is anatomically correct.",
        "",
        f"Patients/seeds evaluated: {scores['pid'].nunique()} patients, {scores['seed'].nunique()} seeds, {len(scores)} patient-seed rows.",
    ]
    (out_dir / "MONITOR_V2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--legacy-monitor", type=Path, required=True)
    ap.add_argument("--endpoint-table", type=Path, required=True)
    ap.add_argument("--spatial-table", type=Path, required=True)
    ap.add_argument("--calibration-table", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260716)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_path = args.endpoint_table
    spatial_path = args.spatial_table
    calibration_path = args.calibration_table
    for path in (endpoint_path, spatial_path, calibration_path, args.legacy_monitor):
        if not path.exists():
            raise FileNotFoundError(path)
    endpoint = pd.read_csv(endpoint_path)
    spatial = pd.read_csv(spatial_path)
    calibration = pd.read_csv(calibration_path)
    novelty = pd.read_csv(args.legacy_monitor)
    scores = build_scores(endpoint, spatial, calibration, novelty, args.seed)
    metrics = build_metrics(scores, args.n_boot, args.seed)
    risk = build_risk_coverage(scores)
    scores_path = args.out_dir / "monitor_v2_scores.csv"
    metrics_path = args.out_dir / "monitor_v2_metrics.csv"
    risk_path = args.out_dir / "monitor_v2_risk_coverage.csv"
    scores.to_csv(scores_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    risk.to_csv(risk_path, index=False)
    make_figure(risk, args.out_dir)
    write_report(metrics, scores, args.out_dir)
    manifest = {
        **C2.script_metadata(Path(__file__)),
        "outputs": {
            "scores": str(scores_path),
            "metrics": str(metrics_path),
            "risk_coverage": str(risk_path),
        },
        "inputs": {
            "endpoint_predictions": str(endpoint_path),
            "spatial_cases": str(spatial_path),
            "calibration_cases": str(calibration_path),
            "legacy_four_architecture_monitor": str(args.legacy_monitor),
        },
        "n_boot": args.n_boot,
        "endpoint_definitions": C2.endpoint_definitions(),
    }
    C2.write_json(args.out_dir / "monitor_v2_manifest.json", manifest)
    print(f"Wrote {scores_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {risk_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
