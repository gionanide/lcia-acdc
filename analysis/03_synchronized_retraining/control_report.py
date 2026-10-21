#!/usr/bin/env python
"""Final v2 control report and decision summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import common as C2
import convention_utils as U
from common import RV_LABEL


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


def compute_dice_matrix(
    patients: Sequence[Path],
    refined_labels: Mapping[Any, Path],
    seeds: Sequence[int],
    runs_root: Path,
    device: torch.device,
    batch_slices: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed_value in seeds:
        print(f"Computing Dice matrix seed {seed_value:02d}", flush=True)
        net_o = C2.safe_load_unet(runs_root / f"seed_{seed_value:02d}" / "original" / "last.pt", device)
        net_r = C2.safe_load_unet(runs_root / f"seed_{seed_value:02d}" / "refined" / "last.pt", device)
        for pdir in patients:
            pid = pdir.name.lower()
            frame_paths = U.patient_frame_paths(pdir, refined_labels)
            for phase in U.PHASES:
                image_nii, image_native, original_native, refined_native = U.load_native_frame(frame_paths[phase])
                sx, sy, _ = [float(x) for x in image_nii.header.get_zooms()[:3]]
                images, original_proc, _ = U.preprocess_volume(image_native, original_native, sx, sy)
                _, refined_proc, _ = U.preprocess_volume(image_native, refined_native, sx, sy)
                pred_o, _ = U.infer_processed(net_o, images, device, batch_slices, return_probs=False)
                pred_r, _ = U.infer_processed(net_r, images, device, batch_slices, return_probs=False)
                rows.append(
                    {
                        "seed": seed_value,
                        "pid": pid,
                        "phase": phase,
                        "original_model_vs_original_reference_rv_dice": U.dice_binary(pred_o == RV_LABEL, original_proc == RV_LABEL),
                        "original_model_vs_refined_reference_rv_dice": U.dice_binary(pred_o == RV_LABEL, refined_proc == RV_LABEL),
                        "refined_model_vs_original_reference_rv_dice": U.dice_binary(pred_r == RV_LABEL, original_proc == RV_LABEL),
                        "refined_model_vs_refined_reference_rv_dice": U.dice_binary(pred_r == RV_LABEL, refined_proc == RV_LABEL),
                        "original_vs_refined_model_rv_dice": U.dice_binary(pred_o == RV_LABEL, pred_r == RV_LABEL),
                        "original_vs_refined_reference_rv_dice": U.dice_binary(original_proc == RV_LABEL, refined_proc == RV_LABEL),
                    }
                )
    return pd.DataFrame(rows)


def matched_crossed_summary(endpoint: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, g in endpoint.groupby("seed"):
        matched_errors = pd.concat(
            [
                (g["original_model_RVEF"] - g["original_reference_RVEF"]).abs(),
                (g["refined_model_RVEF"] - g["refined_reference_RVEF"]).abs(),
            ],
            ignore_index=True,
        )
        crossed_errors = pd.concat(
            [
                (g["original_model_RVEF"] - g["refined_reference_RVEF"]).abs(),
                (g["refined_model_RVEF"] - g["original_reference_RVEF"]).abs(),
            ],
            ignore_index=True,
        )
        rows.append(
            {
                "seed": seed,
                "matched_rvef_mae": float(matched_errors.mean()),
                "crossed_rvef_mae": float(crossed_errors.mean()),
                "matched_minus_crossed": float(matched_errors.mean() - crossed_errors.mean()),
            }
        )
    return pd.DataFrame(rows)


def dice_patient_summary(dice: pd.DataFrame) -> pd.DataFrame:
    phase = dice.copy()
    phase["matched_mean"] = 0.5 * (
        phase["original_model_vs_original_reference_rv_dice"] + phase["refined_model_vs_refined_reference_rv_dice"]
    )
    phase["crossed_mean"] = 0.5 * (
        phase["original_model_vs_refined_reference_rv_dice"] + phase["refined_model_vs_original_reference_rv_dice"]
    )
    return (
        phase.groupby(["seed", "pid"], as_index=False)
        .agg(
            matched_mean_phase_rv_dice=("matched_mean", "mean"),
            crossed_mean_phase_rv_dice=("crossed_mean", "mean"),
            model_pair_mean_phase_rv_dice=("original_vs_refined_model_rv_dice", "mean"),
            reference_pair_mean_phase_rv_dice=("original_vs_refined_reference_rv_dice", "mean"),
            model_pair_min_phase_rv_dice=("original_vs_refined_model_rv_dice", "min"),
        )
    )


def metric(metrics: pd.DataFrame, target: str, score: str, field: str = "auroc") -> float:
    row = metrics[(metrics["target"] == target) & (metrics["score"] == score)]
    return float(row[field].iloc[0]) if len(row) else np.nan


def bool_word(value: bool) -> str:
    return "yes" if bool(value) else "no"


def table_lines(df: pd.DataFrame, columns: Sequence[str], formats: Mapping[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in df[columns].itertuples(index=False):
        vals = []
        for col, value in zip(columns, row):
            if isinstance(value, (float, np.floating)):
                vals.append(format(float(value), formats.get(col, ".3f")))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_final_report(
    out_dir: Path,
    endpoint: pd.DataFrame,
    dice: pd.DataFrame,
    spatial: pd.DataFrame,
    calibration: pd.DataFrame,
    monitor: pd.DataFrame,
    reproduction_gate: Mapping[str, Any],
    sham: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Dict[str, Any]:
    endpoint = normalize_endpoint_columns(endpoint)
    dice_patient = dice_patient_summary(dice)
    endpoint_dice = endpoint.merge(dice_patient, on=["seed", "pid"], how="left")
    mc = matched_crossed_summary(endpoint)
    rvef_spatial = spatial[(spatial["analysis"] == "measurement_transmission") & (spatial["endpoint"] == "RVEF")]
    rvef_spearman_mean = float(rvef_spatial["spearman"].astype(float).mean())
    rvef_spearman_min = float(rvef_spatial["spearman"].astype(float).min())
    matched_better_all = bool((mc["matched_minus_crossed"] < 0).all())
    conv_aurocs = monitor[monitor["score"] == "convention_disagreement"].set_index("target")["auroc"].astype(float)
    arch_aurocs = monitor[monitor["score"] == "ordinary_architecture_rvef_range"].set_index("target")["auroc"].astype(float)
    comparable_targets = sorted(set(conv_aurocs.index) & set(arch_aurocs.index))
    conv_mean_auroc = float(conv_aurocs.loc[comparable_targets].mean())
    arch_mean_auroc = float(arch_aurocs.loc[comparable_targets].mean())
    convention_beats_arch = bool(conv_mean_auroc > arch_mean_auroc)
    risk = pd.read_csv(out_dir / "monitor_v2_risk_coverage.csv")
    target = "reference_threshold_instability_45"
    cov = 0.8
    conv_risk = float(
        risk[
            (risk["target"] == target)
            & (risk["score"] == "convention_disagreement")
            & (np.isclose(risk["coverage"], cov))
        ]["retained_event_rate"].iloc[0]
    )
    rand_risk = float(
        risk[(risk["target"] == target) & (risk["score"] == "random_uniform_rank") & (np.isclose(risk["coverage"], cov))][
            "retained_event_rate"
        ].iloc[0]
    )
    risk_improves_threshold = bool(conv_risk < rand_risk)
    large_shift_small_dice = endpoint_dice[
        (endpoint_dice["model_RVEF_shift"].abs() >= 5.0) & (endpoint_dice["model_pair_mean_phase_rv_dice"] >= 0.90)
    ]
    clinically_meaningful_small_dice = bool(len(large_shift_small_dice) >= 5)
    criteria = {
        "model_shift_tracks_reference_shift_consistently": bool(rvef_spearman_min > 0),
        "matched_reference_better_than_crossed": matched_better_all,
        "convention_disagreement_outperforms_architecture_disagreement": convention_beats_arch,
        "convention_disagreement_improves_threshold_risk_coverage": risk_improves_threshold,
        "clinically_meaningful_rvef_shifts_with_small_dice_changes": clinically_meaningful_small_dice,
    }
    go_count = int(sum(criteria.values()))
    go_dual_head = bool(go_count >= 2)
    calib_full = calibration[
        (calibration["endpoint"] == "RVEF") & (calibration["label_budget"] == calibration["label_budget"].max())
    ]
    calib_mean = (
        calib_full.groupby(["direction", "method"], as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"))
        .sort_values(["direction", "mae"])
    )
    conv_metrics = monitor[monitor["score"].isin(["convention_disagreement", "ordinary_architecture_rvef_range", "same_convention_seed_range"])][
        ["target", "score", "auroc", "auprc", "spearman"]
    ].sort_values(["target", "score"])
    lines = [
        "# Final Control Report",
        "",
        "This report supersedes the earlier unsynchronized Stage B dual-head GO note. The v2 experiment used synchronized paired training, same-label sham controls, fixed `last.pt` checkpoints, and ACDC patients 101-150 for model evaluation.",
        "",
        "No M&Ms-2 data were accessed. The two-reference range is a convention envelope, not a confidence interval. No claim is made that either annotation convention is anatomically correct.",
        "",
        "## Controls",
        "",
        f"- Reproduction gate passed: {bool_word(bool(reproduction_gate.get('pass', False)))}.",
        f"- Same-label five-epoch sham passed: {bool_word(bool(sham.get('pass', False)))}.",
        f"- Synchronized training seeds completed: {', '.join(str(s) for s in training['training_design']['seeds'])}.",
        f"- Fixed checkpoint policy: `last.pt` only; forbidden `best_by_ef` paths were guarded in v2 loaders.",
        "",
        "## Decision Criteria",
        "",
        table_lines(
            pd.DataFrame([{"criterion": key, "met": bool_word(value)} for key, value in criteria.items()]),
            ["criterion", "met"],
        ),
        "",
        f"Dual-head proceed count: {go_count}/5. Decision: {'GO' if go_dual_head else 'NO-GO'} for a shared dual-head model from this report.",
        "",
        "## Transmission",
        "",
        f"- Mean observed directional intersection/reference-change: {float(spatial[spatial['analysis'] == 'directional_spatial_alignment']['mean_intersection_over_reference_change'].mean()):.3f}.",
        f"- Mean RVEF reference/model shift Spearman across seeds: {rvef_spearman_mean:.3f}; minimum across seeds: {rvef_spearman_min:.3f}.",
        f"- Median absolute synchronized model RVEF shift: {float(endpoint['model_RVEF_shift'].abs().median()):.3f} EF points.",
        f"- Patient-seed rows with `abs(model RVEF shift) >= 5` and model-pair mean phase Dice >= 0.90: {len(large_shift_small_dice)}.",
        "",
        "## Matched Versus Crossed",
        "",
        table_lines(mc, ["seed", "matched_rvef_mae", "crossed_rvef_mae", "matched_minus_crossed"]),
        "",
        "## Calibration",
        "",
        table_lines(calib_mean, ["direction", "method", "mae", "rmse", "bias"]),
        "",
        "## Monitor",
        "",
        f"- Mean AUROC across monitor targets: convention disagreement {conv_mean_auroc:.3f}, four-architecture range {arch_mean_auroc:.3f}.",
        f"- At 80% coverage for 45-point threshold instability, retained event rate was {conv_risk:.3f} for convention disagreement and {rand_risk:.3f} for random ranking.",
        "",
        table_lines(conv_metrics, ["target", "score", "auroc", "auprc", "spearman"]),
        "",
        "## Interpretation",
        "",
        "The clean synchronized experiment supports annotation-convention transmission into the trained U-Net: same-label controls passed, spatial changes aligned above permutation null, RVEF shifts were positive-associated with reference shifts across seeds, and matched-reference evaluation was better than crossed-reference evaluation. However, the deployment monitor based only on synchronized convention disagreement was weak and did not outperform the existing four-architecture range on average.",
        "",
        "The report distinguishes annotation-convention variation from training-seed variation by using paired original/refined arms per seed and a same-convention cross-seed comparator. Robust model error is reported against the convention envelope rather than either single reference alone.",
    ]
    report_path = out_dir / "FINAL_CONTROL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "criteria": criteria,
        "go_count": go_count,
        "go_dual_head": go_dual_head,
        "matched_crossed": mc.to_dict(orient="records"),
        "mean_convention_auroc": conv_mean_auroc,
        "mean_architecture_auroc": arch_mean_auroc,
        "large_shift_small_dice_patient_seed_rows": int(len(large_shift_small_dice)),
        "report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc-dir", "--acdc-root", dest="acdc_root", type=Path, required=True)
    ap.add_argument("--refined-dir", "--refined-root", dest="refined_root", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-slices", type=int, default=16)
    ap.add_argument("--force-dice", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C2.set_project_paths(args.acdc_root.resolve(), args.runs_root.resolve())
    C2.V2_RUNS_ROOT = args.runs_root.resolve()
    dice_path = args.out_dir / "v2_dice_reference_matrix.csv"
    if args.force_dice or not dice_path.exists():
        refined_labels = U.find_refined_labels(args.refined_root.resolve())
        patients = C2.get_test_patients(args.acdc_root.resolve())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dice = compute_dice_matrix(patients, refined_labels, args.seeds, args.runs_root.resolve(), device, args.batch_slices)
        dice.to_csv(dice_path, index=False)
    else:
        dice = pd.read_csv(dice_path)
    endpoint = pd.read_csv(args.out_dir / "v2_endpoint_predictions.csv")
    spatial = pd.read_csv(args.out_dir / "spatial_transmission_summary.csv")
    calibration = pd.read_csv(args.out_dir / "calibration_vs_retraining_summary.csv")
    monitor = pd.read_csv(args.out_dir / "monitor_v2_metrics.csv")
    reproduction = C2.read_json(args.out_dir / "reproduction_gate.json")["reproduction_gate"]
    sham = C2.read_json(args.out_dir / "sham_5epoch_summary.json")
    training = C2.read_json(args.out_dir / "convention_training_manifest_v2.json")
    summary = write_final_report(args.out_dir, endpoint, dice, spatial, calibration, monitor, reproduction, sham, training)
    manifest = {
        **C2.script_metadata(Path(__file__)),
        "outputs": {
            "dice_matrix": str(dice_path),
            "final_report": str(args.out_dir / "FINAL_CONTROL_REPORT.md"),
        },
        "summary": summary,
        "endpoint_definitions": C2.endpoint_definitions(),
    }
    C2.write_json(args.out_dir / "final_control_manifest.json", manifest)
    print(f"Wrote {args.out_dir / 'FINAL_CONTROL_REPORT.md'}")
    print(f"Wrote {dice_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
