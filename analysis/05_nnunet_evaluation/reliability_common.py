"""Portable configuration and file identities for the reliability Methods."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path.cwd()
RUN_DIR = Path("outputs")
OUTPUT_DIR = RUN_DIR / "outputs"
SOURCE_RUN_DIR = RUN_DIR
SOURCE_PROJECT = RUN_DIR
SOURCE_CASE_MANIFEST = RUN_DIR / "case_manifest.csv"
INPUT_MANIFEST_PATH = RUN_DIR / "input_manifest.json"
PROTOCOL_PATH = PROJECT_DIR / "config.yaml"
LOCK_PATH = PROTOCOL_PATH
ACDC_ROOT = Path("ACDC")
ACDC_TEST_ROOT = ACDC_ROOT / "testing"
REFINED_ROOT = Path("RefinedLabelsACDC")
MNMS_ROOT = Path("MNM")
MNMS_METADATA = MNMS_ROOT / "metadata.csv"
NNUNET_ROOT = Path("nnunet")
NNUNET_PREPROCESSED_DATASET = NNUNET_ROOT / "nnUNet_preprocessed" / "Dataset701_ACDCDevelopment"
HISTORICAL_CHECKPOINT = Path("last.pt")
QUALIFICATION_DECISION = Path("qualification") / "c2_qualification_decision.json"
CHECKPOINT_RULE = Path("qualification") / "c2_global_checkpoint_rule.json"


def configure(args):
    global RUN_DIR, OUTPUT_DIR, SOURCE_RUN_DIR, SOURCE_PROJECT, SOURCE_CASE_MANIFEST
    global INPUT_MANIFEST_PATH, ACDC_ROOT, ACDC_TEST_ROOT, REFINED_ROOT
    global MNMS_ROOT, MNMS_METADATA, NNUNET_ROOT, NNUNET_PREPROCESSED_DATASET
    global HISTORICAL_CHECKPOINT, QUALIFICATION_DECISION, CHECKPOINT_RULE
    RUN_DIR = args.out_dir.resolve()
    OUTPUT_DIR = RUN_DIR / "outputs"
    SOURCE_RUN_DIR = SOURCE_PROJECT = RUN_DIR
    SOURCE_CASE_MANIFEST = RUN_DIR / "case_manifest.csv"
    INPUT_MANIFEST_PATH = RUN_DIR / "input_manifest.json"
    ACDC_ROOT = args.acdc_dir.resolve()
    ACDC_TEST_ROOT = ACDC_ROOT / "testing"
    REFINED_ROOT = args.refined_dir.resolve()
    MNMS_ROOT = args.mnms_dir.resolve()
    MNMS_METADATA = args.mnms_metadata.resolve()
    NNUNET_ROOT = args.nnunet_root.resolve()
    NNUNET_PREPROCESSED_DATASET = NNUNET_ROOT / "nnUNet_preprocessed" / "Dataset701_ACDCDevelopment"
    HISTORICAL_CHECKPOINT = args.historical_checkpoint.resolve()
    QUALIFICATION_DECISION = args.qualification_dir.resolve() / "c2_qualification_decision.json"
    CHECKPOINT_RULE = args.qualification_dir.resolve() / "c2_global_checkpoint_rule.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_record(path):
    source = Path(path).resolve()
    return {"path": relative_path(source), "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source)}


def read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object")
    return payload


def _json_default(value):
    import numpy as np
    if isinstance(value, Path):
        return relative_path(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    default=_json_default) + "\n", encoding="utf-8")


def load_protocol():
    protocol = OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Unexpected scientific configuration")
    return protocol


def assert_not_forbidden_path(path):
    normalized = str(path).replace("\\", "/").lower()
    if any(token in normalized for token in ("mnm2", "mnms2", "m&ms-2", "m&ms_2", "mms2")):
        raise PermissionError("M&Ms-2 is outside this analysis")


def _validated_path(path, cohort):
    assert_not_forbidden_path(path)
    roots = {"acdc_test": ACDC_TEST_ROOT, "acdc_refined": REFINED_ROOT, "mnms": MNMS_ROOT}
    Path(path).resolve().relative_to(roots[cohort].resolve())


def validated_file(path, *, cohort, purpose):
    _validated_path(path, cohort)
    return file_record(path)


def validated_directory(path, *, cohort, purpose):
    _validated_path(path, cohort)
    if not Path(path).is_dir():
        raise NotADirectoryError(path)


def checkpoint_records():
    payload = read_json(CHECKPOINT_RULE)
    if payload.get("selected_global_rule") != "checkpoint_final":
        raise ValueError("The frozen measurement instrument uses checkpoint_final")
    records = payload["checkpoint_records_by_fold"]
    if set(records) != {str(i) for i in range(5)}:
        raise ValueError("All five folds are required")
    result = []
    for fold in range(5):
        record = records[str(fold)]
        path = Path(record["path"])
        if path.name != "checkpoint_final.pth" or "best_by_ef" in str(path).lower():
            raise ValueError("Unexpected checkpoint rule")
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Checkpoint hash changed")
        result.append({**file_record(path), "fold": fold})
    return result


def checkpoint_paths():
    return [Path(record["path"]) for record in checkpoint_records()]


def verify_qualification():
    payload = read_json(QUALIFICATION_DECISION)
    if payload.get("decision") != "GO_C2_DEVELOPMENT_QUALIFIED" or not all(payload["criteria"].values()):
        raise ValueError("Development qualification has not passed")
    if payload.get("selected_global_rule") != "checkpoint_final":
        raise ValueError("The selected global checkpoint rule changed")
    return payload


def write_status(status, **extra):
    write_json(RUN_DIR / "status.json", {"status": status, "updated_at_utc": now_utc(), **extra})


def records_for_paths(paths):
    return [file_record(path) for path in paths]


def package_versions():
    names = ("torch", "nnunetv2", "monai", "numpy", "pandas", "scipy",
             "statsmodels", "nibabel", "SimpleITK", "matplotlib")
    return {"python": sys.version, **{name: importlib.metadata.version(name) for name in names}}


def set_nnunet_environment():
    os.environ.update(nnUNet_raw=str(NNUNET_ROOT / "nnUNet_raw"),
                      nnUNet_preprocessed=str(NNUNET_ROOT / "nnUNet_preprocessed"),
                      nnUNet_results=str(RUN_DIR / "nnunet_results"), nnUNet_compile="false")
