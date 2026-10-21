#!/usr/bin/env python
"""Phase 1/2 synchronized paired U-Net training and sham controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch

import common as C2
import convention_utils as U
from rng_pairing import (
    model_parameter_max_abs_diff,
    optimizer_state_max_abs_diff,
    paired_train_step,
)


class DeterministicDiceCELoss(torch.nn.Module):
    """Dice + CE loss avoiding CUDA nll_loss2d for synchronized exactness checks."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target_index = target[:, 0].long()
        else:
            target_index = target.long()
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(target_index, num_classes=logits.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)
        dims = tuple(range(2, logits.ndim))
        intersection = torch.sum(probs * one_hot, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(one_hot, dim=dims)
        dice_loss = 1.0 - torch.mean((2.0 * intersection + self.smooth) / (denominator + self.smooth))
        log_probs = torch.log_softmax(logits, dim=1)
        ce = -torch.gather(log_probs, 1, target_index.unsqueeze(1)).mean()
        return dice_loss + ce


def load_phase0_gate(out_dir: Path) -> Dict[str, Any]:
    path = getattr(C2, "REPRODUCTION_GATE", None) or out_dir / "reproduction_gate.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; supply the fixed-model reproduction-gate result.")
    data = C2.read_json(path)
    gate = data["reproduction_gate"]
    if not gate["pass"]:
        raise RuntimeError("Reproduction gate failed; refusing to proceed.")
    if path.resolve() != (out_dir / "reproduction_gate.json").resolve():
        C2.write_json(out_dir / "reproduction_gate.json", data)
    return gate


def load_training_cache(acdc_root: Path, refined_root: Path, split_seed: int) -> tuple[Dict[str, Any], list[Path], list[Path]]:
    C2.set_project_paths(acdc_root, C2.V2_RUNS_ROOT)
    refined_labels = U.find_refined_labels(refined_root)
    train_pat, val_pat = C2.get_train_val_patients(acdc_root, split_seed)
    records = U.build_slice_records(train_pat, refined_labels)
    print(f"train patients={len(train_pat)} val patients={len(val_pat)} train slices={len(records)}", flush=True)
    cache = U.load_cached_training_slices(records)
    return cache, train_pat, val_pat


def order_hash(n: int, seed: int, epochs: int) -> str:
    h = hashlib.sha256()
    for epoch in range(1, epochs + 1):
        h.update(U.epoch_permutation(n, seed, epoch).astype(np.int64).tobytes())
    return h.hexdigest()


def augmentation_hash(cache: Mapping[str, Any], seed: int, epochs: int) -> str:
    h = hashlib.sha256()
    n = int(cache["images"].shape[0])
    for epoch in range(1, epochs + 1):
        order = U.epoch_permutation(n, seed, epoch)
        for idx in order:
            meta = cache["meta"][int(idx)]
            aug = U.deterministic_augmented(
                cache["images"][int(idx)],
                cache["original"][int(idx)],
                cache["refined"][int(idx)],
                seed,
                epoch,
                meta["patient"],
                meta["phase"],
                int(meta["slice_index"]),
            )[3]
            h.update(f"{epoch},{idx},{int(aug['hflip'])},{int(aug['vflip'])}\n".encode("utf-8"))
    return h.hexdigest()


def initial_pair(seed: int, device: torch.device, out_dir: Path) -> tuple[torch.nn.Module, torch.nn.Module, Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    C2.deterministic_setup(seed, strict=True)
    model_a = U.build_model("unet").to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in model_a.state_dict().items()}
    initial_path = out_dir / "initial_state.pt"
    torch.save(initial_state, initial_path)
    model_b = U.build_model("unet").to(device)
    model_a.load_state_dict(initial_state)
    model_b.load_state_dict(initial_state)
    return model_a, model_b, initial_path, C2.sha256_file(initial_path)


def one_batch_sham(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    load_phase0_gate(out_dir)
    cache, train_pat, val_pat = load_training_cache(args.acdc_root.resolve(), args.refined_root.resolve(), args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic = C2.deterministic_setup(args.seed, strict=True)
    model_a, model_b, initial_path, initial_sha = initial_pair(args.seed, device, out_dir / "sham_batch")
    opt_a = torch.optim.Adam(model_a.parameters(), lr=args.lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=args.lr)
    loss_fn = DeterministicDiceCELoss()
    order = U.epoch_permutation(int(cache["images"].shape[0]), args.seed, 1)
    x, y_original, _ = U.batch_from_cache(cache, order[: args.batch_size], args.seed, 1, device)
    result = paired_train_step(model_a, model_b, opt_a, opt_b, loss_fn, x, y_original, y_original, amp=args.amp)
    output = {
        **C2.script_metadata(Path(__file__)),
        "mode": "one_batch_sham",
        "seed": args.seed,
        "device": str(device),
        "deterministic_settings": deterministic,
        "initial_state": str(initial_path),
        "initial_state_sha256": initial_sha,
        "batch_size": args.batch_size,
        "diagnostics": result.__dict__,
        "tolerances": {
            "max_logit_difference": 1e-7,
            "loss_difference": 1e-8,
            "max_gradient_difference": 1e-7,
            "max_parameter_difference": 1e-7,
            "max_optimizer_state_difference": 1e-7,
        },
    }
    output["diagnostics"]["loss_difference"] = abs(result.loss_a - result.loss_b)
    output["pass"] = bool(
        result.max_logit_diff <= 1e-7
        and abs(result.loss_a - result.loss_b) <= 1e-8
        and result.max_gradient_diff <= 1e-7
        and result.max_parameter_diff_after <= 1e-7
        and result.max_optimizer_state_diff_after <= 1e-7
        and result.expected_after_equal
    )
    C2.write_json(out_dir / "sham_batch_test.json", output)
    print(f"One-batch sham pass: {output['pass']}")
    if not output["pass"]:
        raise RuntimeError("One-batch same-label sham failed.")
    return output


def train_pair_core(
    *,
    seed: int,
    label_b: str,
    epochs: int,
    cache: Mapping[str, Any],
    out_root: Path,
    batch_size: int,
    lr: float,
    amp: bool,
    log_every: int,
    device: torch.device,
) -> Dict[str, Any]:
    if label_b not in ("original", "refined"):
        raise ValueError(label_b)
    run_dir = out_root / f"seed_{seed:02d}"
    if label_b == "original":
        arm_a_dir = run_dir / "original_a"
        arm_b_dir = run_dir / "original_b"
    else:
        arm_a_dir = run_dir / "original"
        arm_b_dir = run_dir / label_b
    arm_a_dir.mkdir(parents=True, exist_ok=True)
    arm_b_dir.mkdir(parents=True, exist_ok=True)
    C2.deterministic_setup(seed, strict=True)
    model_a, model_b, initial_path, initial_sha = initial_pair(seed, device, run_dir)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=epochs)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=epochs)
    loss_fn = DeterministicDiceCELoss()
    n = int(cache["images"].shape[0])
    logs: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    paired_failures = 0
    for epoch in range(1, epochs + 1):
        model_a.train()
        model_b.train()
        order = U.epoch_permutation(n, seed, epoch)
        loss_a_sum = 0.0
        loss_b_sum = 0.0
        max_rng_mismatch = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x, y_original, y_refined = U.batch_from_cache(cache, idx, seed, epoch, device)
            y_b = y_original if label_b == "original" else y_refined
            result = paired_train_step(model_a, model_b, opt_a, opt_b, loss_fn, x, y_original, y_b, amp=amp)
            loss_a_sum += result.loss_a
            loss_b_sum += result.loss_b
            if not result.expected_after_equal:
                paired_failures += 1
                max_rng_mismatch += 1
        sched_a.step()
        sched_b.step()
        row = {
            "seed": seed,
            "epoch": epoch,
            "label_a": "original",
            "label_b": label_b,
            "loss_a": loss_a_sum / math.ceil(n / batch_size),
            "loss_b": loss_b_sum / math.ceil(n / batch_size),
            "lr_a": float(sched_a.get_last_lr()[0]),
            "lr_b": float(sched_b.get_last_lr()[0]),
            "parameter_max_abs_diff": model_parameter_max_abs_diff(model_a, model_b),
            "optimizer_state_max_abs_diff": optimizer_state_max_abs_diff(opt_a, opt_b),
            "rng_mismatch_batches": max_rng_mismatch,
            "elapsed_sec": time.perf_counter() - t0,
        }
        logs.append(row)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"seed {seed:02d} {label_b} epoch {epoch:03d}/{epochs} "
                f"loss_a={row['loss_a']:.4f} loss_b={row['loss_b']:.4f} "
                f"param_diff={row['parameter_max_abs_diff']:.3g} elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )
        pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)

    last_a = arm_a_dir / "last.pt"
    last_b = arm_b_dir / "last.pt"
    torch.save(model_a.state_dict(), last_a)
    torch.save(model_b.state_dict(), last_b)
    manifest = {
        "seed": seed,
        "label_a": "original",
        "label_b": label_b,
        "initial_state": str(initial_path),
        "initial_state_sha256": initial_sha,
        "last_a": str(last_a),
        "last_b": str(last_b),
        "last_a_sha256": C2.sha256_file(last_a),
        "last_b_sha256": C2.sha256_file(last_b),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "amp": amp,
        "sample_order_sha256": order_hash(n, seed, epochs),
        "augmentation_sha256": augmentation_hash(cache, seed, epochs),
        "paired_rng_mismatch_batches": paired_failures,
        "training_log": str(run_dir / "training_log.csv"),
        "only_intended_difference": "RV reference label" if label_b == "refined" else "none; same-label sham",
    }
    C2.write_json(run_dir / "seed_manifest.json", manifest)
    return manifest


@torch.inference_mode()
def compare_predictions(a_ckpt: Path, b_ckpt: Path, batch_slices: int, device: torch.device) -> Dict[str, Any]:
    refined_labels = U.find_refined_labels(C2.REFINED_ROOT)
    patients = C2.get_test_patients(C2.ACDC_ROOT)
    net_a = C2.safe_load_unet(a_ckpt, device)
    net_b = C2.safe_load_unet(b_ckpt, device)
    mismatched_voxels = 0
    frames = 0
    rows = []
    for pdir in patients:
        patient_row: Dict[str, Any] = {"pid": pdir.name.lower()}
        frame_paths = U.patient_frame_paths(pdir, refined_labels)
        for phase in U.PHASES:
            image_nii, image_native, original_native, _ = U.load_native_frame(frame_paths[phase])
            sx, sy, sz = [float(x) for x in image_nii.header.get_zooms()[:3]]
            tsx, tsy = U.target_spacing()
            processed_voxel_ml = tsx * tsy * sz / 1000.0
            images, original_processed, _ = U.preprocess_volume(image_native, original_native, sx, sy)
            pred_a, _ = U.infer_processed(net_a, images, device, batch_slices, return_probs=False)
            pred_b, _ = U.infer_processed(net_b, images, device, batch_slices, return_probs=False)
            mismatch = int(np.count_nonzero(pred_a != pred_b))
            mismatched_voxels += mismatch
            frames += 1
            patient_row[f"{phase}_mismatched_voxels"] = mismatch
            patient_row[f"a_{phase}_ml"] = U.volume_ml(pred_a == U.RV_LABEL, processed_voxel_ml)
            patient_row[f"b_{phase}_ml"] = U.volume_ml(pred_b == U.RV_LABEL, processed_voxel_ml)
            patient_row[f"a_dice_{phase}"] = U.dice_binary(pred_a == U.RV_LABEL, original_processed == U.RV_LABEL)
            patient_row[f"b_dice_{phase}"] = U.dice_binary(pred_b == U.RV_LABEL, original_processed == U.RV_LABEL)
        patient_row["a_rvef"] = U.ef(patient_row["a_ED_ml"], patient_row["a_ES_ml"])
        patient_row["b_rvef"] = U.ef(patient_row["b_ED_ml"], patient_row["b_ES_ml"])
        patient_row["abs_rvef_diff"] = abs(patient_row["a_rvef"] - patient_row["b_rvef"])
        rows.append(patient_row)
    df = pd.DataFrame(rows)
    return {
        "n_patients": len(patients),
        "n_frames": frames,
        "mismatched_voxels": mismatched_voxels,
        "predictions_identical": bool(mismatched_voxels == 0),
        "max_abs_rvef_diff": float(df["abs_rvef_diff"].max()),
        "max_abs_edv_diff": float((df["a_ED_ml"] - df["b_ED_ml"]).abs().max()),
        "max_abs_esv_diff": float((df["a_ES_ml"] - df["b_ES_ml"]).abs().max()),
        "max_abs_dice_ed_diff": float((df["a_dice_ED"] - df["b_dice_ED"]).abs().max()),
        "max_abs_dice_es_diff": float((df["a_dice_ES"] - df["b_dice_ES"]).abs().max()),
        "rows": df.to_dict(orient="records"),
    }


def sham_5epoch(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir.resolve()
    load_phase0_gate(out_dir)
    cache, train_pat, val_pat = load_training_cache(args.acdc_root.resolve(), args.refined_root.resolve(), args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root = args.runs_root.resolve() / "sham_5epoch"
    manifest = train_pair_core(
        seed=args.seed,
        label_b="original",
        epochs=5,
        cache=cache,
        out_root=run_root,
        batch_size=args.batch_size,
        lr=args.lr,
        amp=args.amp,
        log_every=1,
        device=device,
    )
    seed_dir = run_root / f"seed_{args.seed:02d}"
    a_ckpt = seed_dir / "original_a" / "last.pt"
    b_ckpt = seed_dir / "original_b" / "last.pt"
    tensor_diff = C2.checkpoint_tensor_diff(a_ckpt, b_ckpt)
    pred_diff = compare_predictions(a_ckpt, b_ckpt, args.batch_slices, device)
    output = {
        **C2.script_metadata(Path(__file__)),
        "mode": "sham_5epoch",
        "seed": args.seed,
        "device": str(device),
        "manifest": manifest,
        "checkpoint_tensor_comparison": tensor_diff,
        "prediction_comparison": pred_diff,
        "pass": bool(
            tensor_diff["bit_identical"]
            and pred_diff["predictions_identical"]
            and manifest["paired_rng_mismatch_batches"] == 0
        ),
    }
    C2.write_json(out_dir / "sham_5epoch_summary.json", output)
    report = out_dir / "SHAM_CONTROL_REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# Sham Control Report",
                "",
                f"One-batch sham: see `{out_dir / 'sham_batch_test.json'}`",
                f"Five-epoch same-label sham pass: `{output['pass']}`",
                f"Checkpoint bit-identical: `{tensor_diff['bit_identical']}`",
                f"Maximum parameter difference: `{tensor_diff['max_abs_parameter_difference']}`",
                f"Predictions identical on ACDC 101-150: `{pred_diff['predictions_identical']}`",
                f"Mismatched voxels: `{pred_diff['mismatched_voxels']}`",
                f"Paired RNG mismatch batches: `{manifest['paired_rng_mismatch_batches']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Five-epoch sham pass: {output['pass']}")
    if not output["pass"]:
        raise RuntimeError("Five-epoch same-label sham failed.")
    return output


def train_convention(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir.resolve()
    load_phase0_gate(out_dir)
    sham_path = out_dir / "sham_5epoch_summary.json"
    if not sham_path.exists() or not C2.read_json(sham_path).get("pass", False):
        raise RuntimeError("Five-epoch sham has not passed; refusing convention training.")
    cache, train_pat, val_pat = load_training_cache(args.acdc_root.resolve(), args.refined_root.resolve(), args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = []
    for seed in args.seeds:
        seed_dir = args.runs_root.resolve() / f"seed_{seed:02d}"
        if (seed_dir / "original" / "last.pt").exists() and (seed_dir / "refined" / "last.pt").exists() and args.skip_existing:
            manifests.append(C2.read_json(seed_dir / "seed_manifest.json"))
            print(f"seed {seed:02d}: skipping existing synchronized checkpoints")
            continue
        manifests.append(
            train_pair_core(
                seed=seed,
                label_b="refined",
                epochs=args.epochs,
                cache=cache,
                out_root=args.runs_root.resolve(),
                batch_size=args.batch_size,
                lr=args.lr,
                amp=args.amp,
                log_every=args.log_every,
                device=device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = {
        **C2.script_metadata(Path(__file__)),
        "mode": "train_convention",
        "acdc_root": str(args.acdc_root.resolve()),
        "refined_root": str(args.refined_root.resolve()),
        "runs_root": str(args.runs_root.resolve()),
        "out_dir": str(out_dir),
        "device": str(device),
        "training_design": {
            "architecture": "unet",
            "seeds": args.seeds,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "amp": args.amp,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "loss": "DiceCELoss(to_onehot_y=True, softmax=True)",
            "loss_implementation": "DeterministicDiceCELoss: explicit softmax Dice plus log_softmax/gather CE",
            "rng_sync": "torch CPU and all CUDA RNG states restored before arm B forward",
            "only_intended_difference": "RV reference label",
        },
        "patient_split": {
            "train": [p.name.lower() for p in train_pat],
            "val": [p.name.lower() for p in val_pat],
        },
        "seed_manifests": manifests,
        "phase0_gate": C2.read_json(out_dir / "reproduction_gate.json")["reproduction_gate"],
        "sham_summary": C2.read_json(out_dir / "sham_5epoch_summary.json"),
    }
    C2.write_json(out_dir / "convention_training_manifest_v2.json", combined)
    print(f"Wrote {out_dir / 'convention_training_manifest_v2.json'}")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sham-batch", "sham-5epoch", "train-convention"], required=True)
    ap.add_argument("--acdc-dir", "--acdc-root", dest="acdc_root", type=Path, required=True)
    ap.add_argument("--refined-dir", "--refined-root", dest="refined_root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--reproduction-gate", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--batch-slices", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    C2.ACDC_ROOT = args.acdc_root.resolve()
    C2.REFINED_ROOT = args.refined_root.resolve()
    C2.V2_CODE_ROOT = args.out_dir.resolve()
    C2.V2_RUNS_ROOT = args.runs_root.resolve()
    C2.REPRODUCTION_GATE = args.reproduction_gate

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "sham-batch":
        one_batch_sham(args)
    elif args.mode == "sham-5epoch":
        sham_5epoch(args)
    elif args.mode == "train-convention":
        train_convention(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
