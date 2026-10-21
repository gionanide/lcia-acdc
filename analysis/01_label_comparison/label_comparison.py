#!/usr/bin/env python
"""Label-comparison Methods: original/revised ACDC masks to geometry CSV/JSON.

Stored-index regional fields retain the historical definition and must not be
interpreted anatomically; the corresponding figure is not a main-paper figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    CONVENTIONS,
    LV_LABEL,
    MYO_LABEL,
    PHASES,
    RV_LABEL,
    all_acdc_patient_dirs,
    basal_zone,
    command_line,
    dice_binary,
    ef,
    find_refined_labels,
    git_version,
    json_default,
    load_native_frame,
    now_utc,
    package_versions,
    patient_frame_paths,
    set_acdc_paths,
    sha256_file,
    volume_ml,
    write_json,
)


EXPECTED_TRANSITIONS = {
    (0, 0),
    (RV_LABEL, RV_LABEL),
    (MYO_LABEL, MYO_LABEL),
    (LV_LABEL, LV_LABEL),
    (0, RV_LABEL),
    (RV_LABEL, 0),
}
ZONES = ("basal", "middle", "apical")


def count_by_zone(mask: np.ndarray) -> Dict[str, int]:
    out = {zone: 0 for zone in ZONES}
    n_slices = mask.shape[2]
    for z in range(n_slices):
        out[basal_zone(z, n_slices)] += int(np.count_nonzero(mask[:, :, z]))
    return out


def transition_matrix(original: np.ndarray, refined: np.ndarray) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.int64)
    for a in range(4):
        for b in range(4):
            mat[a, b] = int(np.count_nonzero((original == a) & (refined == b)))
    return mat


def plot_reference_nesting(rows: pd.DataFrame, out_path: Path, png_metadata: Dict[str, str]) -> None:
    labels = ["original subset", "refined subset", "equal", "neither"]
    counts = []
    for phase in PHASES:
        g = rows[rows.phase == phase]
        equal = (g.original_only_rv_voxels.eq(0) & g.refined_only_rv_voxels.eq(0)).sum()
        original_subset = (g.original_subset_refined_rv & ~g.refined_subset_original_rv).sum()
        refined_subset = (g.refined_subset_original_rv & ~g.original_subset_refined_rv).sum()
        neither = len(g) - int(equal) - int(original_subset) - int(refined_subset)
        counts.append([int(original_subset), int(refined_subset), int(equal), int(neither)])
    arr = np.asarray(counts)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    bottom = np.zeros(len(PHASES))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed"]
    for i, label in enumerate(labels):
        ax.bar(PHASES, arr[:, i], bottom=bottom, label=label, color=colors[i])
        bottom += arr[:, i]
    ax.set_ylabel("Patient-phase count")
    ax.set_title("RV Reference Nesting")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, metadata=png_metadata)
    plt.close(fig)


def plot_change_direction(rows: pd.DataFrame, out_path: Path, png_metadata: Dict[str, str]) -> None:
    labels = []
    original_values = []
    refined_values = []
    for phase in PHASES:
        for zone in ZONES:
            labels.append(f"{phase} {zone}")
            original_values.append(rows[f"original_only_rv_{zone}_voxels"].where(rows.phase == phase, 0).sum())
            refined_values.append(rows[f"refined_only_rv_{zone}_voxels"].where(rows.phase == phase, 0).sum())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar(x - 0.2, original_values, width=0.4, label="original-only RV", color="#dc2626")
    ax.bar(x + 0.2, refined_values, width=0.4, label="refined-only RV", color="#2563eb")
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.set_ylabel("Voxel count")
    ax.set_title("RV Change Direction by Phase and Slice Zone")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, metadata=png_metadata)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc-dir", "--acdc-root", dest="acdc_root", type=Path, required=True)
    ap.add_argument("--refined-dir", "--refined-root", dest="refined_root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-unexpected-class-changes", action="store_true")
    args = ap.parse_args()

    acdc_root = args.acdc_root.resolve()
    refined_root = args.refined_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_acdc_paths(acdc_root)

    refined_labels = find_refined_labels(refined_root)
    patient_dirs = all_acdc_patient_dirs(acdc_root)
    if len(patient_dirs) != 150:
        raise RuntimeError(f"Expected 150 ACDC patients, found {len(patient_dirs)}")

    rows: List[Dict[str, Any]] = []
    total_transition = np.zeros((4, 4), dtype=np.int64)
    changed_patients = set()
    unexpected_rows: List[Dict[str, Any]] = []

    for pdir in patient_dirs:
        pid = pdir.name.lower()
        frames = patient_frame_paths(pdir, refined_labels)
        patient_values: Dict[str, float] = {}
        phase_payload: Dict[str, Dict[str, Any]] = {}
        for phase in PHASES:
            image_nii, _, original, refined = load_native_frame(frames[phase])
            sx, sy, sz = [float(x) for x in image_nii.header.get_zooms()[:3]]
            voxel_ml = sx * sy * sz / 1000.0
            original_rv = original == RV_LABEL
            refined_rv = refined == RV_LABEL

            original_only = original_rv & ~refined_rv
            refined_only = refined_rv & ~original_rv
            intersection = original_rv & refined_rv
            union = original_rv | refined_rv
            if np.any(original != refined):
                changed_patients.add(pid)

            mat = transition_matrix(original, refined)
            total_transition += mat
            unexpected_count = 0
            unexpected_detail = {}
            for a in range(4):
                for b in range(4):
                    if (a, b) not in EXPECTED_TRANSITIONS and mat[a, b] > 0:
                        unexpected_count += int(mat[a, b])
                        unexpected_detail[f"{a}->{b}"] = int(mat[a, b])
            lv_changed = int(np.count_nonzero((original == LV_LABEL) != (refined == LV_LABEL)))
            myo_changed = int(np.count_nonzero((original == MYO_LABEL) != (refined == MYO_LABEL)))
            if unexpected_count or lv_changed or myo_changed:
                unexpected_rows.append(
                    {
                        "patient": pid,
                        "phase": phase,
                        "unexpected_transition_voxels": unexpected_count,
                        "lv_membership_changed_voxels": lv_changed,
                        "myocardium_membership_changed_voxels": myo_changed,
                        "unexpected_transitions_json": json.dumps(unexpected_detail, sort_keys=True),
                    }
                )

            row: Dict[str, Any] = {
                "patient": pid,
                "phase": phase,
                "frame": int(frames[phase]["frame"]),
                "group": frames[phase]["group"],
                "n_slices": int(original.shape[2]),
                "voxel_ml": voxel_ml,
                "original_rv_voxels": int(original_rv.sum()),
                "refined_rv_voxels": int(refined_rv.sum()),
                "original_only_rv_voxels": int(original_only.sum()),
                "refined_only_rv_voxels": int(refined_only.sum()),
                "rv_intersection_voxels": int(intersection.sum()),
                "rv_union_voxels": int(union.sum()),
                "original_subset_refined_rv": bool(np.all(~original_rv | refined_rv)),
                "refined_subset_original_rv": bool(np.all(~refined_rv | original_rv)),
                "reference_rv_dice": dice_binary(original_rv, refined_rv),
                "lv_membership_changed_voxels": lv_changed,
                "myocardium_membership_changed_voxels": myo_changed,
                "unexpected_transition_voxels": unexpected_count,
                "unexpected_transitions_json": json.dumps(unexpected_detail, sort_keys=True),
            }
            for direction, mask in (("original_only_rv", original_only), ("refined_only_rv", refined_only)):
                zone_counts = count_by_zone(mask)
                for zone, count in zone_counts.items():
                    row[f"{direction}_{zone}_voxels"] = int(count)

            patient_values[f"original_{phase.lower()}v_ml"] = volume_ml(original_rv, voxel_ml)
            patient_values[f"refined_{phase.lower()}v_ml"] = volume_ml(refined_rv, voxel_ml)
            phase_payload[phase] = row

        original_edv = patient_values["original_edv_ml"]
        refined_edv = patient_values["refined_edv_ml"]
        original_esv = patient_values["original_esv_ml"]
        refined_esv = patient_values["refined_esv_ml"]
        original_rvef = ef(original_edv, original_esv)
        refined_rvef = ef(refined_edv, refined_esv)
        patient_level = {
            "original_edv_ml": original_edv,
            "refined_edv_ml": refined_edv,
            "delta_edv_ml": refined_edv - original_edv,
            "original_esv_ml": original_esv,
            "refined_esv_ml": refined_esv,
            "delta_esv_ml": refined_esv - original_esv,
            "original_rvef": original_rvef,
            "refined_rvef": refined_rvef,
            "delta_rvef": refined_rvef - original_rvef,
            "abs_delta_rvef": abs(refined_rvef - original_rvef),
        }
        for phase in PHASES:
            row = dict(phase_payload[phase])
            row.update(patient_level)
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "convention_label_geometry.csv"
    summary_path = out_dir / "convention_label_geometry_summary.json"
    nesting_path = out_dir / "figure_reference_nesting.png"
    direction_path = out_dir / "figure_change_direction_by_phase.png"

    metadata = {
        "created_utc": now_utc(),
        "command": command_line(),
        "code_version": git_version(),
        "package_versions": package_versions(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "acdc_root": str(acdc_root),
        "refined_root": str(refined_root),
        "inputs": {
            "refined_root": str(refined_root),
        },
        "terminology": {
            "reference_names": list(CONVENTIONS),
            "two_reference_range": "convention envelope",
        },
        "expected_transitions": sorted([f"{a}->{b}" for a, b in EXPECTED_TRANSITIONS]),
        "slice_zone_definition": "stored slice index: lower third basal, middle third middle, upper third apical",
    }
    transition_dict = {
        f"{a}->{b}": int(total_transition[a, b])
        for a in range(4)
        for b in range(4)
        if total_transition[a, b] > 0
    }
    summary = {
        "metadata": metadata,
        "n_patients": len(patient_dirs),
        "n_patient_phase_rows": len(df),
        "changed_patients": len(changed_patients),
        "rv": {
            "patient_phase_original_subset_refined": int(df["original_subset_refined_rv"].sum()),
            "patient_phase_refined_subset_original": int(df["refined_subset_original_rv"].sum()),
            "median_reference_rv_dice": float(df["reference_rv_dice"].median()),
            "mean_reference_rv_dice": float(df["reference_rv_dice"].mean()),
            "total_original_only_rv_voxels": int(df["original_only_rv_voxels"].sum()),
            "total_refined_only_rv_voxels": int(df["refined_only_rv_voxels"].sum()),
        },
        "endpoint_changes": {
            "median_abs_delta_rvef": float(df.drop_duplicates("patient")["abs_delta_rvef"].median()),
            "n_abs_delta_rvef_gt_5": int((df.drop_duplicates("patient")["abs_delta_rvef"] > 5).sum()),
            "mean_delta_edv_ml": float(df.drop_duplicates("patient")["delta_edv_ml"].mean()),
            "mean_delta_esv_ml": float(df.drop_duplicates("patient")["delta_esv_ml"].mean()),
        },
        "class_change_verification": {
            "lv_membership_changed_voxels": int(df["lv_membership_changed_voxels"].sum()),
            "myocardium_membership_changed_voxels": int(df["myocardium_membership_changed_voxels"].sum()),
            "unexpected_transition_voxels": int(df["unexpected_transition_voxels"].sum()),
            "transition_matrix": transition_dict,
            "unexpected_rows": unexpected_rows,
        },
        "outputs": {
            "csv": str(csv_path),
            "summary": str(summary_path),
            "figure_reference_nesting": str(nesting_path),
            "figure_change_direction_by_phase": str(direction_path),
        },
    }

    df.to_csv(csv_path, index=False)
    write_json(summary_path, summary)
    png_metadata = {
        "Title": "ACDC annotation-convention label geometry",
        "Description": json.dumps(metadata, default=json_default, sort_keys=True),
    }
    plot_reference_nesting(df, nesting_path, png_metadata)
    plot_change_direction(df, direction_path, png_metadata)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {nesting_path}")
    print(f"Wrote {direction_path}")
    print(
        "Class-change verification: "
        f"LV={summary['class_change_verification']['lv_membership_changed_voxels']} "
        f"MYO={summary['class_change_verification']['myocardium_membership_changed_voxels']} "
        f"unexpected={summary['class_change_verification']['unexpected_transition_voxels']}"
    )

    if (
        not args.allow_unexpected_class_changes
        and summary["class_change_verification"]["unexpected_transition_voxels"] > 0
    ):
        raise RuntimeError("Unexpected non-RV class transitions detected; stopping before training.")
    if (
        not args.allow_unexpected_class_changes
        and (
            summary["class_change_verification"]["lv_membership_changed_voxels"] > 0
            or summary["class_change_verification"]["myocardium_membership_changed_voxels"] > 0
        )
    ):
        raise RuntimeError("LV or myocardium membership changed; stopping before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
