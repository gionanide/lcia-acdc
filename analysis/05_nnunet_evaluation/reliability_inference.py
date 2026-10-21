"""Frozen C2 fold inference, official probability ensemble, and historical comparator."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

import reliability_common as common
from reliability_data import load_case_manifest
from reliability_metrics import mean_pairwise_absolute, measurement_from_phase_volumes, phase_volume_ml


PREDICTION_ROOT = common.RUN_DIR / "predictions"
C2_ROOT = PREDICTION_ROOT / "c2"
HISTORICAL_ROOT = PREDICTION_ROOT / "historical_unet"
INFERENCE_MANIFEST_PATH = common.OUTPUT_DIR / "reliability_inference_manifest.json"
FOLD_MEASUREMENTS_PATH = common.RUN_DIR / "fold_measurements.csv"
FOLD_DISAGREEMENT_PATH = common.RUN_DIR / "fold_disagreement.csv"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def _build_predictor(checkpoint_path: Path):
    """Exact qualified OOF predictor construction, extended only to save probabilities."""
    common.set_nnunet_environment()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = _read_json(common.NNUNET_PREPROCESSED_DATASET / "nnUNetPlans.json")
    dataset = _read_json(common.NNUNET_PREPROCESSED_DATASET / "dataset.json")
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint.get("network_weights"), dict):
        raise RuntimeError("Invalid official nnU-Net checkpoint weights")
    configuration_name = str(checkpoint["init_args"]["configuration"])
    if configuration_name != "2d" or str(checkpoint["trainer_name"]) != "QualificationNNUNetTrainer":
        raise RuntimeError("Checkpoint does not match the frozen qualified C2 contract")
    if int(checkpoint["current_epoch"]) != 1000:
        raise RuntimeError("C2 checkpoint is not the fixed 1,000-epoch final checkpoint")
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    label_manager = plans_manager.get_label_manager(dataset)
    input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset)
    network = nnUNetTrainer.build_network_architecture(
        plans_manager,
        configuration_manager,
        input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    network.load_state_dict(checkpoint["network_weights"], strict=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.manual_initialization(
        network,
        plans_manager,
        configuration_manager,
        [checkpoint["network_weights"]],
        dataset,
        str(checkpoint["trainer_name"]),
        checkpoint.get("inference_allowed_mirroring_axes"),
    )
    contract = {
        "configuration": "2d",
        "trainer": "QualificationNNUNetTrainer",
        "checkpoint_current_epoch": 1000,
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "perform_everything_on_device": True,
        "save_probabilities": True,
        "num_processes_preprocessing": 2,
        "num_processes_segmentation_export": 2,
        "native_grid_export": True,
    }
    return predictor, contract


def _prediction_inventory(directory: Path, identifiers: list[str], *, probabilities: bool) -> dict[str, Any]:
    suffixes = [".nii.gz"] + ([".npz", ".pkl"] if probabilities else [])
    records: list[dict[str, Any]] = []
    for identifier in identifiers:
        for suffix in suffixes:
            records.append(common.file_record(directory / f"{identifier}{suffix}"))
    digest = __import__("hashlib").sha256()
    for record in records:
        digest.update(f"{record['path']}\0{record['sha256']}\n".encode("utf-8"))
    return {
        "directory": common.relative_path(directory),
        "case_count": len(identifiers),
        "file_count": len(records),
        "aggregate_sha256": digest.hexdigest(),
        "files": records,
    }


def _run_fold(
    fold: int, checkpoint_path: Path, cases: pd.DataFrame, identifiers: list[str]
) -> dict[str, Any]:
    output_dir = C2_ROOT / f"fold_{fold}"
    manifest_path = output_dir / "fold_inference_manifest.json"
    if output_dir.exists():
        raise RuntimeError(f"Fold output already exists; automatic retry is forbidden: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    predictor = None
    started = time.perf_counter()
    started_at = common.now_utc()
    try:
        predictor, contract = _build_predictor(checkpoint_path)
        inputs = [[str((common.WORKSPACE / path).resolve())] for path in cases["image_path"]]
        outputs = [str((output_dir / identifier).resolve()) for identifier in identifiers]
        predictor.predict_from_files(
            inputs,
            outputs,
            save_probabilities=True,
            overwrite=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )
        torch.cuda.synchronize()
        inventory = _prediction_inventory(output_dir, identifiers, probabilities=True)
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "fold": fold,
            "started_at_utc": started_at,
            "ended_at_utc": common.now_utc(),
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": common.file_record(checkpoint_path),
            "predictor_contract": contract,
            "prediction_inventory": inventory,
            "training": False,
            "postprocessing": False,
            "calibration": False,
        }
        common.write_json(manifest_path, manifest)
        return manifest
    finally:
        if predictor is not None:
            predictor.network = None
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_official_probability_ensemble(identifiers: list[str]) -> dict[str, Any]:
    from nnunetv2.ensembling.ensemble import ensemble_folders

    output_dir = C2_ROOT / "ensemble"
    if output_dir.exists():
        raise RuntimeError("C2 ensemble output already exists; automatic replacement is forbidden")
    started = time.perf_counter()
    ensemble_folders(
        [str((C2_ROOT / f"fold_{fold}").resolve()) for fold in range(5)],
        str(output_dir.resolve()),
        save_merged_probabilities=True,
        num_processes=2,
        dataset_json_file_or_dict=str(
            (common.NNUNET_PREPROCESSED_DATASET / "dataset.json").resolve()
        ),
        plans_json_file_or_dict=str(
            (common.NNUNET_PREPROCESSED_DATASET / "nnUNetPlans.json").resolve()
        ),
    )
    return {
        "implementation": "nnunetv2.ensembling.ensemble.ensemble_folders",
        "arithmetic": "float32 mean of five native-grid fold probability arrays",
        "segmentation": "official label-manager probability conversion",
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_inventory": _prediction_inventory(output_dir, identifiers, probabilities=True),
    }


def _save_native_prediction(source: nib.Nifti1Image, prediction: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(prediction.astype(np.uint8), source.affine, header=header)
    qform, qcode = source.get_qform(coded=True)
    sform, scode = source.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode or 0))
    if sform is not None:
        output.set_sform(sform, int(scode or 0))
    nib.save(output, str(path))


def _historical_prediction(cases: pd.DataFrame, identifiers: list[str]) -> dict[str, Any]:
    if HISTORICAL_ROOT.exists():
        raise RuntimeError("Historical prediction output exists; automatic replacement is forbidden")
    shared = Path(__file__).resolve().parents[1] / "_shared"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    from convention_utils import (
        preprocess_volume,
        processed_prediction_to_native,
    )
    from model_zoo import build_model

    device = torch.device("cuda")
    network = build_model("unet").to(device)
    state = torch.load(common.HISTORICAL_CHECKPOINT, map_location=device, weights_only=True)
    network.load_state_dict(state, strict=True)
    network.eval()
    started = time.perf_counter()
    output_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row, identifier in zip(cases.to_dict("records"), identifiers):
            image_path = common.WORKSPACE / str(row["image_path"])
            image_nii = nib.load(str(image_path.resolve()))
            image = np.asarray(image_nii.dataobj, dtype=np.float32)
            if image.ndim != 3 or not np.isfinite(image).all():
                raise RuntimeError(f"Invalid staged image for historical U-Net: {identifier}")
            sx, sy = (float(value) for value in image_nii.header.get_zooms()[:2])
            processed, _, resampled_shapes = preprocess_volume(
                image, np.zeros_like(image, dtype=np.int16), sx, sy
            )
            batches: list[np.ndarray] = []
            for start in range(0, len(processed), 8):
                tensor = torch.from_numpy(processed[start : start + 8]).unsqueeze(1).to(device)
                logits = network(tensor)
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError(f"Historical U-Net nonfinite logits: {identifier}")
                batches.append(logits.argmax(dim=1).cpu().numpy().astype(np.int16))
            processed_prediction = np.concatenate(batches, axis=0)
            native = processed_prediction_to_native(
                processed_prediction,
                resampled_shapes=resampled_shapes,
                native_shape=tuple(int(v) for v in image.shape),
            )
            output_path = HISTORICAL_ROOT / f"{identifier}.nii.gz"
            _save_native_prediction(image_nii, native, output_path)
            reference_path = common.WORKSPACE / str(row["original_reference_path"])
            prediction_sitk = sitk.ReadImage(str(output_path.resolve()))
            reference_sitk = sitk.ReadImage(str(reference_path.resolve()))
            if (
                prediction_sitk.GetSize() != reference_sitk.GetSize()
                or not np.allclose(prediction_sitk.GetSpacing(), reference_sitk.GetSpacing(), rtol=0, atol=1e-6)
                or not np.allclose(prediction_sitk.GetOrigin(), reference_sitk.GetOrigin(), rtol=0, atol=1e-5)
                or not np.allclose(prediction_sitk.GetDirection(), reference_sitk.GetDirection(), rtol=0, atol=1e-6)
            ):
                raise RuntimeError(f"Historical native projection geometry mismatch: {identifier}")
            output_records.append(common.file_record(output_path))
    del network
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "implementation": "MONAI_UNet_fixed_SHA256_identified_historical_checkpoint",
        "checkpoint": common.file_record(common.HISTORICAL_CHECKPOINT),
        "preprocessing": "dataset.py via validated convention_utils.preprocess_volume",
        "prediction_to_native": "validated convention_utils.processed_prediction_to_native",
        "batch_slices": 8,
        "postprocessing": False,
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_count": len(output_records),
        "predictions": output_records,
    }


def _fold_measurement_outputs(cases: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        for (cohort, patient_id), group in cases.groupby(["cohort", "patient_id"], sort=True):
            if set(group["phase"]) != {"ED", "ES"}:
                raise RuntimeError(f"Incomplete fold measurement pair: {patient_id}")
            identifiers = group.set_index("phase")["identifier"]
            edv = phase_volume_ml(C2_ROOT / f"fold_{fold}" / f"{identifiers['ED']}.nii.gz")
            esv = phase_volume_ml(C2_ROOT / f"fold_{fold}" / f"{identifiers['ES']}.nii.gz")
            endpoints = measurement_from_phase_volumes(edv, esv)
            first = group.iloc[0]
            rows.append(
                {
                    "cohort": cohort,
                    "patient_id": patient_id,
                    "vendor": first["vendor"],
                    "pathology": first["pathology"],
                    "fold": fold,
                    **endpoints,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["cohort", "patient_id", "fold"])
    frame.to_csv(FOLD_MEASUREMENTS_PATH, index=False)
    disagreement_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(["cohort", "patient_id", "vendor", "pathology"], sort=True):
        values = group.sort_values("fold")["rvef_points"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        disagreement_rows.append(
            {
                "cohort": key[0],
                "patient_id": key[1],
                "vendor": key[2],
                "pathology": key[3],
                "finite_fold_count": int(len(finite)),
                "fold_rvef_range_points": float(np.ptp(finite)) if len(finite) else float("nan"),
                "fold_rvef_standard_deviation_points_ddof1": (
                    float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
                ),
                "mean_pairwise_absolute_fold_rvef_difference_points": mean_pairwise_absolute(values),
                **{f"fold_{fold}_rvef_points": float(values[fold]) for fold in range(5)},
            }
        )
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(FOLD_DISAGREEMENT_PATH, index=False)
    return {
        "fold_measurements": common.file_record(FOLD_MEASUREMENTS_PATH),
        "fold_disagreement": common.file_record(FOLD_DISAGREEMENT_PATH),
        "patient_count": int(disagreement["patient_id"].nunique()),
    }


def run_all_inference() -> dict[str, Any]:
    common.load_protocol()
    common.verify_qualification()
    if INFERENCE_MANIFEST_PATH.exists() or C2_ROOT.exists() or HISTORICAL_ROOT.exists():
        raise RuntimeError("Inference artifacts already exist; automatic retry or overwrite is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen reliability inference stage")
    cases = load_case_manifest().sort_values(["cohort", "patient_id", "phase"]).reset_index(drop=True)
    identifiers = cases["identifier"].astype(str).tolist()
    checkpoints = common.checkpoint_paths()
    common.write_status(
        "inference_running",
        started_at_utc=common.now_utc(),
        case_count=len(cases),
        fold_count=5,
    )
    fold_manifests: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for fold, checkpoint in enumerate(checkpoints):
            fold_manifests.append(_run_fold(fold, checkpoint, cases, identifiers))
        ensemble = _run_official_probability_ensemble(identifiers)
        historical = _historical_prediction(cases, identifiers)
        fold_outputs = _fold_measurement_outputs(cases)
        manifest = {
            "schema_version": 1,
            "decision": "FROZEN_RELIABILITY_INFERENCE_COMPLETE",
            "generated_at_utc": common.now_utc(),
            "elapsed_seconds": time.perf_counter() - started,
            "case_count": len(cases),
            "patient_count": int(cases[["cohort", "patient_id"]].drop_duplicates().shape[0]),
            "fold_manifests": fold_manifests,
            "official_probability_ensemble": ensemble,
            "historical_comparator": historical,
            "fold_disagreement_outputs": fold_outputs,
            "training": False,
            "calibration": False,
            "postprocessing": False,
            "checkpoint_rule": "checkpoint_final_for_all_five_folds",
            "no_MnMs2_access": True,
        }
        common.write_json(INFERENCE_MANIFEST_PATH, manifest)
        common.write_status(
            "inference_completed",
            inference_manifest=common.file_record(INFERENCE_MANIFEST_PATH),
        )
        return manifest
    except BaseException as exc:
        common.write_status(
            "inference_failed_terminal",
            error_type=type(exc).__name__,
            error=str(exc),
            automatic_retry=False,
        )
        raise
