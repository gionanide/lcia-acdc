#!/usr/bin/env python
"""Phase 3 spatial convention-transmission audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import common as C2
import convention_utils as U
from common import RV_LABEL


def support_bounds(mask_zxy: np.ndarray) -> Dict[str, Any]:
    # mask_zxy is processed prediction/reference in (Z,H,W).
    present = np.where(mask_zxy.reshape(mask_zxy.shape[0], -1).any(axis=1))[0]
    if len(present) == 0:
        return {"support_length": 0, "basal_boundary": np.nan, "apical_boundary": np.nan}
    return {
        "support_length": int(len(present)),
        "basal_boundary": int(present.min()),
        "apical_boundary": int(present.max()),
    }


def directional_metrics(ref_mask: np.ndarray, model_mask: np.ndarray) -> Dict[str, float]:
    ref = ref_mask.astype(bool, copy=False)
    model = model_mask.astype(bool, copy=False)
    inter = int(np.count_nonzero(ref & model))
    ref_n = int(np.count_nonzero(ref))
    model_n = int(np.count_nonzero(model))
    total = int(ref.size)
    precision = inter / model_n if model_n else np.nan
    recall = inter / ref_n if ref_n else np.nan
    dice = 2 * inter / (ref_n + model_n) if (ref_n + model_n) else np.nan
    frac_inside = precision
    ref_fraction = ref_n / total if total else np.nan
    enrichment = frac_inside / ref_fraction if np.isfinite(frac_inside) and ref_fraction > 0 else np.nan
    return {
        "reference_change_voxels": ref_n,
        "model_change_voxels": model_n,
        "intersection_voxels": inter,
        "directional_dice": dice,
        "precision": precision,
        "recall": recall,
        "intersection_over_reference_change": recall,
        "fraction_model_change_inside_reference_change": frac_inside,
        "reference_region_fraction": ref_fraction,
        "enrichment": enrichment,
    }


def zone_counts(mask_zxy: np.ndarray) -> Dict[str, int]:
    out = {"basal": 0, "middle": 0, "apical": 0}
    n = mask_zxy.shape[0]
    for z in range(n):
        out[U.basal_zone(z, n)] += int(np.count_nonzero(mask_zxy[z]))
    return out


@torch.inference_mode()
def evaluate_seed(seed: int, patients: list[Path], refined_labels, device: torch.device, batch_slices: int):
    original_ckpt = C2.V2_RUNS_ROOT / f"seed_{seed:02d}" / "original" / "last.pt"
    refined_ckpt = C2.V2_RUNS_ROOT / f"seed_{seed:02d}" / "refined" / "last.pt"
    net_o = C2.safe_load_unet(original_ckpt, device)
    net_r = C2.safe_load_unet(refined_ckpt, device)
    frame_maps: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    case_rows: List[Dict[str, Any]] = []
    endpoint_rows: List[Dict[str, Any]] = []
    for pdir in patients:
        pid = pdir.name.lower()
        frames = U.patient_frame_paths(pdir, refined_labels)
        endpoint: Dict[str, Any] = {"seed": seed, "pid": pid}
        for phase in U.PHASES:
            image_nii, image_native, original_native, refined_native = U.load_native_frame(frames[phase])
            sx, sy, sz = [float(x) for x in image_nii.header.get_zooms()[:3]]
            native_voxel_ml = sx * sy * sz / 1000.0
            images, original_proc, shapes = U.preprocess_volume(image_native, original_native, sx, sy)
            _, refined_proc, _ = U.preprocess_volume(image_native, refined_native, sx, sy)
            pred_o_proc, prob_o = U.infer_processed(net_o, images, device, batch_slices, return_probs=True)
            pred_r_proc, prob_r = U.infer_processed(net_r, images, device, batch_slices, return_probs=True)
            pred_o_nat = U.processed_prediction_to_native(pred_o_proc, shapes, tuple(image_native.shape))
            pred_r_nat = U.processed_prediction_to_native(pred_r_proc, shapes, tuple(image_native.shape))

            o_ref = original_proc == RV_LABEL
            r_ref = refined_proc == RV_LABEL
            o_pred = pred_o_proc == RV_LABEL
            r_pred = pred_r_proc == RV_LABEL
            g_add = r_ref & ~o_ref
            g_remove = o_ref & ~r_ref
            p_add = r_pred & ~o_pred
            p_remove = o_pred & ~r_pred
            delta_p = prob_r[:, RV_LABEL] - prob_o[:, RV_LABEL]

            endpoint[f"reference_{phase}_shift_ml"] = U.volume_ml(refined_native == RV_LABEL, native_voxel_ml) - U.volume_ml(original_native == RV_LABEL, native_voxel_ml)
            endpoint[f"model_{phase}_shift_ml"] = U.volume_ml(pred_r_nat == RV_LABEL, native_voxel_ml) - U.volume_ml(pred_o_nat == RV_LABEL, native_voxel_ml)
            endpoint[f"original_reference_{phase}_ml"] = U.volume_ml(original_native == RV_LABEL, native_voxel_ml)
            endpoint[f"refined_reference_{phase}_ml"] = U.volume_ml(refined_native == RV_LABEL, native_voxel_ml)
            endpoint[f"original_model_{phase}_ml"] = U.volume_ml(pred_o_nat == RV_LABEL, native_voxel_ml)
            endpoint[f"refined_model_{phase}_ml"] = U.volume_ml(pred_r_nat == RV_LABEL, native_voxel_ml)
            endpoint[f"original_model_entropy_{phase}"] = U.mean_entropy(prob_o)
            endpoint[f"refined_model_entropy_{phase}"] = U.mean_entropy(prob_r)

            support_o = support_bounds(o_pred)
            support_r = support_bounds(r_pred)
            frame_maps[(seed, pid, phase)] = {
                "g_add": g_add,
                "g_remove": g_remove,
                "p_add": p_add,
                "p_remove": p_remove,
                "o_pred": o_pred,
                "r_pred": r_pred,
                "delta_p": delta_p,
            }
            for direction, g_mask, p_mask in (("add", g_add, p_add), ("remove", g_remove, p_remove)):
                row = {
                    "seed": seed,
                    "pid": pid,
                    "phase": phase,
                    "direction": direction,
                    "soft_mass_inside_reference_change": float(delta_p[g_mask].sum()) if np.count_nonzero(g_mask) else 0.0,
                    "soft_mass_outside_reference_change": float(delta_p[~g_mask].sum()),
                    "support_length_original_model": support_o["support_length"],
                    "support_length_refined_model": support_r["support_length"],
                    "support_length_shift": support_r["support_length"] - support_o["support_length"],
                    "basal_boundary_shift": support_r["basal_boundary"] - support_o["basal_boundary"] if np.isfinite(support_o["basal_boundary"]) and np.isfinite(support_r["basal_boundary"]) else np.nan,
                    "apical_boundary_shift": support_r["apical_boundary"] - support_o["apical_boundary"] if np.isfinite(support_o["apical_boundary"]) and np.isfinite(support_r["apical_boundary"]) else np.nan,
                    "rv_presence_disagreement_slices": int(np.count_nonzero(o_pred.reshape(o_pred.shape[0], -1).any(axis=1) != r_pred.reshape(r_pred.shape[0], -1).any(axis=1))),
                }
                row.update(directional_metrics(g_mask, p_mask))
                for prefix, mask in (("reference", g_mask), ("model", p_mask)):
                    for zone, count in zone_counts(mask).items():
                        row[f"{prefix}_{zone}_voxels"] = count
                case_rows.append(row)
        endpoint["reference_EDV_shift"] = endpoint["reference_ED_shift_ml"]
        endpoint["reference_ESV_shift"] = endpoint["reference_ES_shift_ml"]
        endpoint["model_EDV_shift"] = endpoint["model_ED_shift_ml"]
        endpoint["model_ESV_shift"] = endpoint["model_ES_shift_ml"]
        endpoint["original_reference_rvef"] = U.ef(endpoint["original_reference_ED_ml"], endpoint["original_reference_ES_ml"])
        endpoint["refined_reference_rvef"] = U.ef(endpoint["refined_reference_ED_ml"], endpoint["refined_reference_ES_ml"])
        endpoint["original_model_rvef"] = U.ef(endpoint["original_model_ED_ml"], endpoint["original_model_ES_ml"])
        endpoint["refined_model_rvef"] = U.ef(endpoint["refined_model_ED_ml"], endpoint["refined_model_ES_ml"])
        endpoint["reference_RVEF_shift"] = endpoint["refined_reference_rvef"] - endpoint["original_reference_rvef"]
        endpoint["model_RVEF_shift"] = endpoint["refined_model_rvef"] - endpoint["original_model_rvef"]
        endpoint_rows.append(endpoint)
    return pd.DataFrame(case_rows), pd.DataFrame(endpoint_rows), frame_maps


def correlation_row(df: pd.DataFrame, seed: Any, endpoint: str) -> Dict[str, Any]:
    ref = df[f"reference_{endpoint}_shift"]
    model = df[f"model_{endpoint}_shift"]
    reg = U.regression_stats(ref, model)
    valid = np.isfinite(ref) & np.isfinite(model) & (np.abs(ref) > 1e-8)
    sign = float((np.sign(ref[valid]) == np.sign(model[valid])).mean()) if valid.any() else np.nan
    return {
        "seed": seed,
        "analysis": "measurement_transmission",
        "endpoint": endpoint,
        **reg,
        "sign_agreement": sign,
        "median_abs_model_shift": float(model.abs().median()),
        "mean_reference_shift": float(ref.mean()),
        "mean_model_shift": float(model.mean()),
        "transmission_fraction_mean": float(model.mean() / ref.mean()) if abs(float(ref.mean())) > 1e-8 else np.nan,
    }


def permutation_null(frame_maps: Dict[Tuple[int, str, str], Dict[str, Any]], n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for seed_value in sorted({k[0] for k in frame_maps}):
        for phase in U.PHASES:
            pids = sorted({k[1] for k in frame_maps if k[0] == seed_value and k[2] == phase})
            for direction in ("add", "remove"):
                g_key = f"g_{direction}"
                p_key = f"p_{direction}"
                obs_vals = []
                for pid in pids:
                    ref = frame_maps[(seed_value, pid, phase)][g_key]
                    model = frame_maps[(seed_value, pid, phase)][p_key]
                    ref_n = np.count_nonzero(ref)
                    if ref_n:
                        obs_vals.append(np.count_nonzero(ref & model) / ref_n)
                observed = float(np.mean(obs_vals)) if obs_vals else np.nan
                perm_vals = []
                shape_groups: Dict[Tuple[int, ...], List[str]] = {}
                for pid in pids:
                    shape = tuple(frame_maps[(seed_value, pid, phase)][g_key].shape)
                    shape_groups.setdefault(shape, []).append(pid)
                for perm_id in range(n_perm):
                    vals = []
                    for grouped_pids in shape_groups.values():
                        if len(grouped_pids) < 2:
                            continue
                        shuffled = rng.permutation(grouped_pids)
                        for ref_pid, model_pid in zip(grouped_pids, shuffled):
                            ref = frame_maps[(seed_value, ref_pid, phase)][g_key]
                            model = frame_maps[(seed_value, model_pid, phase)][p_key]
                            ref_n = np.count_nonzero(ref)
                            if ref_n:
                                vals.append(np.count_nonzero(ref & model) / ref_n)
                    perm_vals.append(float(np.mean(vals)) if vals else np.nan)
                arr = np.asarray(perm_vals, dtype=float)
                finite_arr = arr[np.isfinite(arr)]
                rows.append(
                    {
                        "null": "patient_permutation",
                        "seed": seed_value,
                        "phase": phase,
                        "direction": direction,
                        "metric": "intersection_over_reference_change",
                        "observed": observed,
                        "null_mean": float(np.mean(finite_arr)) if finite_arr.size else np.nan,
                        "null_ci_low": float(np.percentile(finite_arr, 2.5)) if finite_arr.size else np.nan,
                        "null_ci_high": float(np.percentile(finite_arr, 97.5)) if finite_arr.size else np.nan,
                        "p_ge_observed": float(np.mean(finite_arr >= observed))
                        if finite_arr.size and np.isfinite(observed)
                        else np.nan,
                        "n_perm": n_perm,
                        "n_shape_groups": len(shape_groups),
                    }
                )
    return pd.DataFrame(rows)


def same_convention_null(frame_maps: Dict[Tuple[int, str, str], Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    seeds = sorted({k[0] for k in frame_maps})
    for a_i, seed_a in enumerate(seeds):
        for seed_b in seeds[a_i + 1 :]:
            for phase in U.PHASES:
                pids = sorted({k[1] for k in frame_maps if k[0] == seed_a and k[2] == phase})
                for direction in ("add", "remove"):
                    vals = []
                    for pid in pids:
                        ref = frame_maps[(seed_a, pid, phase)][f"g_{direction}"]
                        pred_a = frame_maps[(seed_a, pid, phase)]["o_pred"]
                        pred_b = frame_maps[(seed_b, pid, phase)]["o_pred"]
                        model = (pred_b & ~pred_a) if direction == "add" else (pred_a & ~pred_b)
                        ref_n = np.count_nonzero(ref)
                        if ref_n:
                            vals.append(np.count_nonzero(ref & model) / ref_n)
                    rows.append(
                        {
                            "null": "same_convention_cross_seed",
                            "seed": f"{seed_a}_vs_{seed_b}",
                            "phase": phase,
                            "direction": direction,
                            "metric": "intersection_over_reference_change",
                            "observed": np.nan,
                            "null_mean": float(np.mean(vals)) if vals else np.nan,
                            "null_ci_low": np.nan,
                            "null_ci_high": np.nan,
                            "p_ge_observed": np.nan,
                            "n_perm": 0,
                        }
                    )
    return pd.DataFrame(rows)


def make_figures(cases: pd.DataFrame, endpoint: pd.DataFrame, nulls: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=160)
    plot = cases.groupby(["phase", "direction"], as_index=False)["intersection_over_reference_change"].mean()
    labels = [f"{r.phase} {r.direction}" for r in plot.itertuples()]
    ax.bar(labels, plot["intersection_over_reference_change"], color="#2563eb")
    ax.set_ylabel("Mean intersection / reference-change voxels")
    ax.set_title("Directional Spatial Alignment")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_directional_alignment.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=160)
    for phase, g in cases.groupby("phase"):
        ax.scatter(g["support_length_shift"], g["rv_presence_disagreement_slices"], label=phase, alpha=0.7)
    ax.set_xlabel("Predicted RV support length shift")
    ax.set_ylabel("Slices with RV-presence disagreement")
    ax.set_title("Reference/Model Support Shift")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_reference_model_support_shift.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=160)
    ax.scatter(endpoint["reference_EDV_shift"], endpoint["model_EDV_shift"], label="EDV", alpha=0.7)
    ax.scatter(endpoint["reference_ESV_shift"], endpoint["model_ESV_shift"], label="ESV", alpha=0.7)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Reference volume shift (mL)")
    ax.set_ylabel("Model volume shift (mL)")
    ax.set_title("EDV/ESV Transmission")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_edv_esv_transmission.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, nulls: pd.DataFrame, out_dir: Path) -> None:
    spatial = summary[summary.analysis == "directional_spatial_alignment"]
    meas = summary[summary.analysis == "measurement_transmission"]
    rvef = meas[meas.endpoint == "RVEF"]
    edv = meas[meas.endpoint == "EDV"]
    esv = meas[meas.endpoint == "ESV"]
    perm = nulls[nulls.null == "patient_permutation"]
    lines = [
        "# Spatial Transmission Report",
        "",
        "This report tests whether synchronized original-vs-refined model changes align with original-reference versus refined-reference RV changes.",
        "",
        "## Directional Spatial Alignment",
        f"- Mean observed intersection/reference-change: `{spatial['mean_intersection_over_reference_change'].mean():.4f}`",
        f"- Mean patient-permutation null: `{perm['null_mean'].mean():.4f}`",
        "",
        "## Measurement Transmission",
        f"- EDV Spearman mean across seeds: `{edv['spearman'].mean():.3f}`",
        f"- ESV Spearman mean across seeds: `{esv['spearman'].mean():.3f}`",
        f"- RVEF Spearman mean across seeds: `{rvef['spearman'].mean():.3f}`",
        "",
        "## Interpretation",
        "Spatial alignment and EDV/ESV/RVEF transmission should be interpreted against the permutation and same-convention null rows in `spatial_transmission_nulls.csv`.",
    ]
    (out_dir / "SPATIAL_TRANSMISSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--acdc-dir", type=Path, required=True)
    ap.add_argument("--refined-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-slices", type=int, default=16)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C2.V2_RUNS_ROOT = args.runs_root.resolve()
    C2.ACDC_ROOT = args.acdc_dir.resolve()
    C2.REFINED_ROOT = args.refined_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patients = C2.get_test_patients(C2.ACDC_ROOT)
    refined_labels = U.find_refined_labels(C2.REFINED_ROOT)
    all_cases = []
    all_endpoints = []
    all_maps = {}
    for seed_value in args.seeds:
        print(f"Evaluating spatial transmission seed {seed_value:02d}", flush=True)
        cases, endpoints, maps = evaluate_seed(seed_value, patients, refined_labels, device, args.batch_slices)
        all_cases.append(cases)
        all_endpoints.append(endpoints)
        all_maps.update(maps)
    cases = pd.concat(all_cases, ignore_index=True)
    endpoint = pd.concat(all_endpoints, ignore_index=True)
    endpoint_path = args.out_dir / "v2_endpoint_predictions.csv"
    endpoint.to_csv(endpoint_path, index=False)

    summary_rows = []
    for (seed_value, phase, direction), g in cases.groupby(["seed", "phase", "direction"]):
        summary_rows.append(
            {
                "analysis": "directional_spatial_alignment",
                "seed": seed_value,
                "phase": phase,
                "direction": direction,
                "mean_directional_dice": float(g["directional_dice"].mean()),
                "mean_precision": float(g["precision"].mean()),
                "mean_recall": float(g["recall"].mean()),
                "mean_intersection_over_reference_change": float(g["intersection_over_reference_change"].mean()),
                "mean_fraction_model_change_inside_reference_change": float(g["fraction_model_change_inside_reference_change"].mean()),
                "mean_enrichment": float(g["enrichment"].replace([np.inf, -np.inf], np.nan).mean()),
            }
        )
    for seed_value, g in endpoint.groupby("seed"):
        for endpoint_name in ("EDV", "ESV", "RVEF"):
            summary_rows.append(correlation_row(g, seed_value, endpoint_name))
    patient_mean = endpoint.groupby("pid", as_index=False).agg(
        reference_EDV_shift=("reference_EDV_shift", "first"),
        model_EDV_shift=("model_EDV_shift", "mean"),
        reference_ESV_shift=("reference_ESV_shift", "first"),
        model_ESV_shift=("model_ESV_shift", "mean"),
        reference_RVEF_shift=("reference_RVEF_shift", "first"),
        model_RVEF_shift=("model_RVEF_shift", "mean"),
    )
    for endpoint_name in ("EDV", "ESV", "RVEF"):
        summary_rows.append(correlation_row(patient_mean, "patient_clustered_mean_across_seeds", endpoint_name))
    summary = pd.DataFrame(summary_rows)

    nulls = pd.concat(
        [
            permutation_null(all_maps, args.n_perm, args.seed),
            same_convention_null(all_maps),
        ],
        ignore_index=True,
    )

    cases_path = args.out_dir / "spatial_transmission_cases.csv"
    summary_path = args.out_dir / "spatial_transmission_summary.csv"
    nulls_path = args.out_dir / "spatial_transmission_nulls.csv"
    cases.to_csv(cases_path, index=False)
    summary.to_csv(summary_path, index=False)
    nulls.to_csv(nulls_path, index=False)
    make_figures(cases, endpoint, nulls, args.out_dir)
    write_report(summary, nulls, args.out_dir)
    manifest = {
        **C2.script_metadata(Path(__file__)),
        "outputs": {
            "spatial_transmission_cases": str(cases_path),
            "spatial_transmission_summary": str(summary_path),
            "spatial_transmission_nulls": str(nulls_path),
            "v2_endpoint_predictions": str(endpoint_path),
        },
        "n_perm": args.n_perm,
        "seeds": args.seeds,
        "device": str(device),
        "endpoint_definitions": C2.endpoint_definitions(),
    }
    C2.write_json(args.out_dir / "spatial_transmission_manifest.json", manifest)
    print(f"Wrote {cases_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {nulls_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
