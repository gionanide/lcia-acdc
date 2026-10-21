"""nnU-Net qualification Methods: completed folds to native OOF metrics and rule.

Uses both global checkpoint rules, complete development patients and the
separate nonselectable fold-0 continuation. No clinical endpoint selects a rule.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["nnUNet_preprocessed"] = str(args.nnunet_preprocessed.resolve())
    os.environ["nnUNet_raw"] = str(args.raw_dataset_dir.resolve().parent)
    os.environ["nnUNet_results"] = str(out / "inference_contract")
    import numpy as np
    import pandas as pd
    import training_io
    training_io.PREPROCESSED_DATASET = args.nnunet_preprocessed / "Dataset701_ACDCDevelopment"
    from oof_predictor import build_predictor
    from oof_metrics import evaluate_group, aggregate_patients
    from oof_selection import (
        RULES, summarize_global_rules, select_global_rule, paired_rule_comparisons,
        bca_interval,
    )
    splits = json.loads(Path(__file__).with_name("development_splits.json").read_text())
    cases, patient_frames, records = [], [], {}
    finite = True
    continuation_patients = None
    for fold in range(5):
        fold_dir = args.training_dir / "folds" / f"fold_{fold}"
        manifest = training_io.read_json(fold_dir / "run_manifest.json")
        audit = pd.read_csv(fold_dir / "base_epoch_audit.csv")
        if manifest["status"] != "completed" or audit.completed_epoch.tolist() != list(range(1, 1001)):
            raise ValueError(f"Incomplete official fold {fold}")
        finite &= bool(audit.finite_epoch.all())
        finite &= all(r["all_finite"] for r in manifest["base"]["checkpoints"].values())
        identifiers = splits[fold]["val"]
        for identifier in identifiers:
            training_io.assert_development_case(identifier)
        rules = list(RULES) + (["continuation_nonselectable"] if fold == 0 else [])
        for rule in rules:
            if rule == "continuation_nonselectable":
                rec = manifest["continuation"]["checkpoints"]["checkpoint_final"]
                continuation_audit = pd.read_csv(fold_dir / "continuation_epoch_audit.csv")
                if continuation_audit.completed_epoch.tolist() != list(range(1001, 1201)):
                    raise ValueError("Incomplete fold-0 continuation")
                finite &= bool(continuation_audit.finite_epoch.all()) and rec["all_finite"]
            else:
                rec = manifest["base"]["checkpoints"][rule]
            ckpt = Path(rec["path"])
            if training_io.sha256_file(ckpt) != rec["sha256"]:
                raise ValueError("Checkpoint identity changed")
            predictor, _ = build_predictor(ckpt)
            folder = out / "oof" / rule / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=False)
            predictor.predict_from_files(
                [[str((args.raw_dataset_dir / "imagesTr" / f"{case}_0000.nii.gz").resolve())]
                 for case in identifiers],
                [str((folder / case).resolve()) for case in identifiers],
                save_probabilities=False, overwrite=False,
                num_processes_preprocessing=2, num_processes_segmentation_export=2)
            frame = evaluate_group(folder, args.raw_dataset_dir / "labelsTr",
                                   case_identifiers=identifiers, fold=fold, checkpoint_rule=rule)
            patients = aggregate_patients(frame)
            cases.append(frame)
            if rule == "continuation_nonselectable":
                continuation_patients = patients.sort_values("patient")
            else:
                patient_frames.append(patients)
                records.setdefault(rule, {})[str(fold)] = rec
            del predictor
    patients = pd.concat(patient_frames, ignore_index=True)
    summary = summarize_global_rules(patients)
    selection = select_global_rule(summary)
    selected = summary.set_index("checkpoint_rule").loc[selection["selected_global_rule"]]
    base = patients[(patients.checkpoint_rule == "checkpoint_final") & (patients.fold == 0)].sort_values("patient")
    if base.patient.tolist() != continuation_patients.patient.tolist() or len(base) != 20:
        raise ValueError("Continuation patient pairing changed")
    continuation = {}
    for offset, (column, limit) in enumerate((("rv_dice_mean_phase", 0.002), ("rv_dice_minimum_phase", 0.003))):
        delta = continuation_patients[column].to_numpy() - base[column].to_numpy()
        low, high = bca_interval(delta, n_resamples=5000, seed=20260811 + 500 + offset)
        continuation[column] = {"gain": float(delta.mean()), "limit": limit,
                                "pass": bool(delta.mean() <= limit), "bca95_low": low, "bca95_high": high}
    safety = all(float(summary[column].max()) - float(selected[column]) <= 0.005
                 for column in ("mean_lv_dice", "mean_myocardium_dice"))
    criteria = {"all_folds_finite": bool(finite),
                "continuation": all(item["pass"] for item in continuation.values()),
                "lv_myocardium_safety": safety,
                "complete_oof": all(len(patients[patients.checkpoint_rule == rule]) == 100 for rule in RULES),
                "zero_invalid": bool((summary.invalid_patient_rate == 0).all())}
    decision = {"decision": "GO_C2_DEVELOPMENT_QUALIFIED" if all(criteria.values()) else "NO_GO_C2_DEVELOPMENT_NOT_QUALIFIED",
                "criteria": criteria, "selected_global_rule": selection["selected_global_rule"]}
    pd.concat(cases).to_csv(out / "oof_case_metrics.csv", index=False)
    patients.to_csv(out / "oof_patient_segmentation_metrics.csv", index=False)
    summary.to_csv(out / "c2_oof_rule_summary.csv", index=False)
    paired_rule_comparisons(patients, n_resamples=5000, seed=20260811).to_csv(out / "c2_oof_paired_comparison.csv", index=False)
    (out / "c2_continuation_audit.json").write_text(json.dumps(continuation, indent=2))
    (out / "c2_qualification_decision.json").write_text(json.dumps(decision, indent=2))
    selection.update(decision="C2_GLOBAL_CHECKPOINT_RULE_FROZEN",
                     checkpoint_records_by_fold=records[selection["selected_global_rule"]])
    (out / "c2_global_checkpoint_rule.json").write_text(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
