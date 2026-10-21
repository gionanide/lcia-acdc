"""nnU-Net training Methods: prepared development data to fixed fold checkpoints.

Run one 1,000-epoch official fold. Fold 0 additionally runs the separate
200-epoch nonselectable continuation. No pretrained checkpoint is selected.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.out_dir.resolve()
    if (root / "folds" / f"fold_{args.fold}").exists():
        raise FileExistsError("Fold exists; use a fresh destination, not resume")
    os.environ.update(nnUNet_raw=str(args.nnunet_raw.resolve()),
                      nnUNet_preprocessed=str(args.nnunet_preprocessed.resolve()),
                      nnUNet_results=str(root / "official_results"),
                      nnUNet_compile="false", nnUNet_n_proc_DA="8")
    import numpy as np
    import torch
    from training_core import (
        QualificationNNUNetTrainer, QualificationNNUNetContinuationTrainer,
        audit_checkpoint, close_training_resources,
    )
    from training_io import file_record

    if not torch.cuda.is_available():
        raise RuntimeError("The frozen training configuration requires CUDA")
    seed = 20260804 + args.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data_dir = args.nnunet_preprocessed / "Dataset701_ACDCDevelopment"
    plans = json.loads((data_dir / "nnUNetPlans.json").read_text())
    supplied_plans = json.loads(Path(__file__).with_name("nnUNetPlans.json").read_text())
    if plans != supplied_plans:
        raise ValueError("Preprocessed plans do not equal the supplied study plans")
    splits = json.loads((data_dir / "splits_final.json").read_text())
    frozen_splits = json.loads(Path(__file__).with_name("development_splits.json").read_text())
    if splits != frozen_splits:
        raise ValueError("Patient-level development split changed")
    dataset = json.loads((data_dir / "dataset.json").read_text())
    fold_dir = root / "folds" / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)

    def train(continuation=False, source=None):
        cls = QualificationNNUNetContinuationTrainer if continuation else QualificationNNUNetTrainer
        stage_plans = copy.deepcopy(plans)
        stage_plans["continue_training"] = continuation
        trainer = cls(stage_plans, "2d", args.fold, dataset,
                      device=torch.device("cuda"))
        if continuation:
            trainer.output_folder = str(root / "continuation" / "fold_0")
            trainer.output_folder_base = str(root / "continuation")
            Path(trainer.output_folder).mkdir(parents=True, exist_ok=False)
            trainer.log_file = str(Path(trainer.output_folder) / "training_log.txt")
        try:
            trainer.initialize()
            if continuation:
                trainer.load_checkpoint(str(source))
                if trainer.current_epoch != 1000:
                    raise ValueError("Continuation must start from epoch 1,000")
                trainer.num_epochs = 1200
                trainer.initial_lr = 0.0001
                for group in trainer.optimizer.param_groups:
                    group["lr"] = group["initial_lr"] = 0.0001
                trainer.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    trainer.optimizer, factor=1.0, total_iters=200)
            elif trainer.num_epochs != 1000:
                raise ValueError("Official 1,000-epoch schedule changed")
            cfg = trainer.configuration_manager
            if cfg.batch_size != 56 or list(cfg.patch_size) != [256, 224]:
                raise ValueError("Frozen batch or patch size changed")
            if trainer.num_iterations_per_epoch != 250 or trainer.num_val_iterations_per_epoch != 50:
                raise ValueError("Official epoch iteration counts changed")
            stage = "continuation" if continuation else "base"
            trainer.configure_qualification_audit(
                path=fold_dir / f"{stage}_epoch_audit.csv", stage=stage,
                stage_started=time.perf_counter())
            trainer.run_training()
            folder = Path(trainer.output_folder)
            final = audit_checkpoint(folder / "checkpoint_final.pth",
                                     expected_epoch=1200 if continuation else 1000,
                                     selectable=not continuation)
            result = {"status": "completed", "seed": seed, "split": splits[args.fold],
                      "checkpoints": {"checkpoint_final": final}}
            if not continuation:
                result["checkpoints"]["checkpoint_best"] = audit_checkpoint(
                    folder / "checkpoint_best.pth", selectable=True)
            return result
        finally:
            close_training_resources(trainer)

    base = train()
    continuation = None
    if args.fold == 0:
        continuation = train(True, Path(base["checkpoints"]["checkpoint_final"]["path"]))
    manifest = {"status": "completed", "fold": args.fold, "seed": seed,
                "base": base, "continuation": continuation,
                "plans": file_record(data_dir / "nnUNetPlans.json")}
    (fold_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
