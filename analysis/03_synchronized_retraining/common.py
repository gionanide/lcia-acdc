#!/usr/bin/env python
"""Shared utilities for convention-monitor v2 controls."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED = PROJECT_ROOT / "analysis" / "_shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import convention_utils as U


V2_CODE_ROOT = Path("outputs")
V2_RUNS_ROOT = Path("checkpoints")
ACDC_ROOT = Path("ACDC")
REFINED_ROOT = Path("RefinedLabelsACDC")
OLD_CODE_ROOT = Path("supporting_inputs")
OLD_RUNS_ROOT = Path("historical_checkpoints")

RV_LABEL = U.RV_LABEL
PHASES = U.PHASES
THRESHOLDS = U.THRESHOLDS
FAILURE_TOLERANCE = U.FAILURE_TOLERANCE


def ensure_dirs() -> None:
    V2_CODE_ROOT.mkdir(parents=True, exist_ok=True)
    V2_RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    return U.sha256_file(path, chunk_size)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_default(value: Any) -> Any:
    return U.json_default(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    U.write_json(path, payload)


def git_version(cwd: Path = PROJECT_ROOT) -> Dict[str, Any]:
    return U.git_version(cwd)


def package_versions() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "monai": U.optional_version("monai"),
        "nibabel": U.nib.__version__,
    }


def command_line() -> str:
    return " ".join([Path(sys.executable).name, *sys.argv])


def script_metadata(script_path: Path) -> Dict[str, Any]:
    return {
        "script": str(script_path.resolve()),
        "script_sha256": sha256_file(script_path.resolve()),
        "created_utc": now_utc(),
        "command": command_line(),
        "code_version": git_version(),
        "package_versions": package_versions(),
    }


def deterministic_setup(seed: int, strict: bool = True) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    deterministic_algorithms = None
    deterministic_error = None
    warn_only = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_algorithms = True
            warn_only = True
        except Exception as exc:
            deterministic_algorithms = False
            deterministic_error = str(exc)
    return {
        "seed": seed,
        "random_seed_set": True,
        "numpy_seed_set": True,
        "torch_seed_set": True,
        "cuda_seed_set": bool(torch.cuda.is_available()),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": deterministic_algorithms,
        "deterministic_algorithms_warn_only": warn_only,
        "deterministic_error": deterministic_error,
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checkpoint_tensor_diff(a_path: Path, b_path: Path) -> Dict[str, Any]:
    a = torch.load(a_path, map_location="cpu", weights_only=True)
    b = torch.load(b_path, map_location="cpu", weights_only=True)
    keys = sorted(set(a) | set(b))
    max_abs = 0.0
    differing = 0
    missing = []
    for key in keys:
        if key not in a or key not in b:
            missing.append(key)
            continue
        diff = (a[key] - b[key]).abs()
        d = float(diff.max().item()) if diff.numel() else 0.0
        if d > 0:
            differing += 1
        max_abs = max(max_abs, d)
    return {
        "state_dict_sha256_a": state_dict_sha256(a),
        "state_dict_sha256_b": state_dict_sha256(b),
        "bit_identical": state_dict_sha256(a) == state_dict_sha256(b),
        "max_abs_parameter_difference": max_abs,
        "n_differing_tensors": differing,
        "missing_keys": missing,
    }


def hash_rng_state() -> Dict[str, Any]:
    out = {
        "cpu": sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        "cuda": [],
    }
    if torch.cuda.is_available():
        out["cuda"] = [
            sha256_bytes(state.cpu().numpy().tobytes())
            for state in torch.cuda.get_rng_state_all()
        ]
    return out


def set_project_paths(acdc_root: Path = ACDC_ROOT, runs_root: Optional[Path] = None) -> None:
    U.set_acdc_paths(acdc_root, runs_root)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_definitions() -> Dict[str, Any]:
    return {
        "statistical_unit": "patient",
        "volume_and_rvef": "native-grid prediction masks and native-grid original/refined references",
        "dice_and_spatial": "processed-grid masks",
        "two_reference_range": "convention envelope",
        "checkpoint_selection": "fixed end-of-schedule last.pt only",
        "forbidden_checkpoint_pattern": "best_by_ef",
        "reference_shift": "refined reference value minus original reference value",
        "model_shift": "refined-label trained model value minus original-label trained model value",
    }


def fail_on_forbidden_checkpoint(path: Path) -> None:
    if "best_by_ef" in path.name.lower() or "best_by_ef" in str(path.parent).lower():
        raise RuntimeError(f"Refusing forbidden checkpoint path: {path}")


def safe_load_unet(path: Path, device: torch.device) -> torch.nn.Module:
    fail_on_forbidden_checkpoint(path)
    return U.load_unet_checkpoint(path, device)


def get_test_patients(acdc_root: Path = ACDC_ROOT) -> list[Path]:
    set_project_paths(acdc_root)
    patients = [Path(p) for p in U.list_patients(Path(acdc_root) / "testing")]
    if len(patients) != 50:
        raise RuntimeError(f"Expected 50 ACDC test patients, found {len(patients)}")
    return patients


def get_train_val_patients(acdc_root: Path = ACDC_ROOT, split_seed: int = 42) -> tuple[list[Path], list[Path]]:
    set_project_paths(acdc_root)
    train, val = U.stratified_split(U.C.TRAIN_DIR, U.C.VAL_FRACTION, split_seed)
    return [Path(p) for p in train], [Path(p) for p in val]


def copy_specs_manifest() -> Dict[str, Any]:
    """Return the public scientific configuration identity, if present."""
    path = Path(__file__).with_name("config.yaml")
    return {path.name: {"sha256": sha256_file(path)}} if path.is_file() else {}


def run_subprocess(args: Sequence[str], cwd: Path = PROJECT_ROOT) -> Dict[str, Any]:
    proc = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, check=False)
    return {
        "args": list(args),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
