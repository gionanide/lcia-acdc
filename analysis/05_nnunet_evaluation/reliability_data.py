"""Post-lock cohort validation and native-grid staging."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.linalg import polar

import reliability_common as common
from reliability_metrics import (
    canonicalize_scaled_labels,
    geometry_matches,
    measurement_from_phase_volumes,
    native_volume_ml,
    remap_mnms_to_instrument,
)


STAGING_DIR = common.SOURCE_RUN_DIR / "staging"
CASE_MANIFEST_PATH = common.SOURCE_CASE_MANIFEST
GATE_PATH = common.SOURCE_PROJECT / "outputs" / "restricted_cohort_validation_gate.json"
MNMS_SUBSETS = (Path("Training") / "Labeled", Path("Validation"), Path("Testing"))
DIRECTION_GRAM_TOLERANCE = 1e-6
MAXIMUM_CORNER_DISPLACEMENT_MM = 0.1
MAXIMUM_CORNER_DISPLACEMENT_FRACTION_OF_MINIMUM_SPACING = 0.05
MAXIMUM_FLOAT32_SERIALIZATION_CORNER_DISPLACEMENT_MM = 0.001


def _read_info(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    if "ED" not in info or "ES" not in info:
        raise RuntimeError(f"Missing ED/ES in {path}")
    return info


def _sitk_read(path: Path) -> sitk.Image:
    previous = sitk.ProcessObject.GetGlobalWarningDisplay()
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    try:
        return sitk.ReadImage(str(path.resolve()))
    finally:
        sitk.ProcessObject.SetGlobalWarningDisplay(previous)


def _load_integer_label(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path.resolve()))
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all() or not np.array_equal(data, np.rint(data)):
        raise RuntimeError(f"Invalid integer label volume: {path}")
    labels = np.rint(data).astype(np.int16, copy=False)
    unexpected = set(np.unique(labels).tolist()) - {0, 1, 2, 3}
    if unexpected:
        raise RuntimeError(f"Unexpected labels {sorted(unexpected)}: {path}")
    return image, labels


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if common.sha256_file(destination) != expected_sha256:
        raise RuntimeError(f"Staged copy changed bytes: {source}")


def _find_refined_labels() -> dict[tuple[int, int], Path]:
    common.validated_directory(
        common.REFINED_ROOT, cohort="acdc_refined", purpose="discover_refined_reference_files"
    )
    patient_pattern = re.compile(r"patient(\d{3})", re.IGNORECASE)
    frame_pattern = re.compile(r"(?:frame|fr)[_\-]?(\d{1,3})", re.IGNORECASE)
    result: dict[tuple[int, int], Path] = {}
    for path in common.REFINED_ROOT.rglob("*"):
        if not path.is_file() or not (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")):
            continue
        patient_match = patient_pattern.search(str(path))
        frame_match = frame_pattern.search(path.name)
        if patient_match is None or frame_match is None:
            continue
        key = (int(patient_match.group(1)), int(frame_match.group(1)))
        if key in result:
            raise RuntimeError(f"Duplicate refined reference for patient/frame {key}")
        result[key] = path
    return result


def _acdc_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    refined = _find_refined_labels()
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    lv_changed = 0
    myocardium_changed = 0
    for patient_number in range(101, 151):
        patient_id = f"patient{patient_number:03d}"
        patient_dir = common.ACDC_TEST_ROOT / patient_id
        info_path = patient_dir / "Info.cfg"
        records.append(common.validated_file(info_path, cohort="acdc_test", purpose="read_ED_ES_metadata"))
        info = _read_info(info_path)
        if int(info["ED"]) == int(info["ES"]):
            raise RuntimeError(f"ACDC ED and ES are identical for {patient_id}")
        for phase in ("ED", "ES"):
            frame = int(info[phase])
            image_path = patient_dir / f"{patient_id}_frame{frame:02d}.nii.gz"
            original_path = patient_dir / f"{patient_id}_frame{frame:02d}_gt.nii.gz"
            refined_path = refined.get((patient_number, frame))
            if refined_path is None:
                raise FileNotFoundError(f"Missing refined reference for {patient_id} frame {frame}")
            image_record = common.validated_file(
                image_path, cohort="acdc_test", purpose="stage_ACDC_test_image"
            )
            original_record = common.validated_file(
                original_path, cohort="acdc_test", purpose="stage_original_reference"
            )
            refined_record = common.validated_file(
                refined_path, cohort="acdc_refined", purpose="stage_refined_reference"
            )
            records.extend((image_record, original_record, refined_record))
            original_nii, original = _load_integer_label(original_path)
            refined_nii, refined_labels = _load_integer_label(refined_path)
            if original.shape != refined_labels.shape:
                raise RuntimeError(f"Original/refined shape mismatch for {patient_id} {phase}")
            if not geometry_matches(_sitk_read(original_path), _sitk_read(refined_path)):
                raise RuntimeError(f"Original/refined native geometry mismatch for {patient_id} {phase}")
            if not geometry_matches(_sitk_read(image_path), _sitk_read(original_path)):
                raise RuntimeError(f"Image/reference native geometry mismatch for {patient_id} {phase}")
            if not np.array_equal(original == 3, refined_labels == 3):
                lv_changed += 1
            if not np.array_equal(original == 2, refined_labels == 2):
                myocardium_changed += 1
            if lv_changed or myocardium_changed:
                raise RuntimeError("Unexpected LV or myocardium change in refined ACDC references")
            if not np.allclose(original_nii.affine, refined_nii.affine, rtol=0, atol=1e-5):
                raise RuntimeError(f"Original/refined NIfTI affine mismatch for {patient_id} {phase}")

            identifier = f"ACDC_{patient_number:03d}_{phase}"
            staged_image = STAGING_DIR / "acdc" / "images" / f"{identifier}_0000.nii.gz"
            staged_original = STAGING_DIR / "acdc" / "references" / "original" / f"{identifier}.nii.gz"
            staged_refined = STAGING_DIR / "acdc" / "references" / "refined" / f"{identifier}.nii.gz"
            _copy_verified(image_path, staged_image, image_record["sha256"])
            _copy_verified(original_path, staged_original, original_record["sha256"])
            _copy_verified(refined_path, staged_refined, refined_record["sha256"])
            rows.append(
                {
                    "cohort": "ACDC_test",
                    "patient_id": patient_id,
                    "patient_number": patient_number,
                    "vendor": "ACDC",
                    "vendor_name": "ACDC",
                    "pathology": info.get("Group", "unknown"),
                    "phase": phase,
                    "frame_index": frame,
                    "identifier": identifier,
                    "image_path": common.relative_path(staged_image),
                    "original_reference_path": common.relative_path(staged_original),
                    "refined_reference_path": common.relative_path(staged_refined),
                }
            )
    checks = {
        "patient_count": len({row["patient_id"] for row in rows}) == 50,
        "case_count": len(rows) == 100,
        "complete_phase_pairs": all(
            set(group["phase"]) == {"ED", "ES"}
            for _, group in pd.DataFrame(rows).groupby("patient_id")
        ),
        "lv_unchanged": lv_changed == 0,
        "myocardium_unchanged": myocardium_changed == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"ACDC restricted-cohort gate failed: {checks}")
    return rows, records, checks


def _strict_frame(value: Any, *, code: str, name: str, n_frames: int) -> int:
    number = float(value)
    if not np.isfinite(number) or number != int(number):
        raise RuntimeError(f"Invalid {name} frame for {code}: {value}")
    frame = int(number)
    if frame < 0 or frame >= n_frames:
        raise RuntimeError(f"Out-of-bounds {name} frame for {code}: {frame}/{n_frames}")
    return frame


def _save_extracted_nifti(
    source: nib.Nifti1Image,
    data: np.ndarray,
    destination: Path,
    dtype: np.dtype,
    *,
    affine: np.ndarray | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = source.header.copy()
    header.set_data_dtype(dtype)
    chosen_affine = np.asarray(source.affine if affine is None else affine, dtype=np.float64)
    output = nib.Nifti1Image(np.asarray(data, dtype=dtype), chosen_affine, header=header)
    output.set_qform(chosen_affine, 1)
    output.set_sform(chosen_affine, 1)
    nib.save(output, str(destination))


def _bounded_orthonormal_affine(source: nib.Nifti1Image) -> tuple[np.ndarray, dict[str, Any]]:
    affine = np.asarray(source.affine, dtype=np.float64)
    columns = affine[:3, :3]
    spacing = np.linalg.norm(columns, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise RuntimeError("Invalid NIfTI affine spacing for direction audit")
    direction = columns / spacing
    gram_error = float(np.max(np.abs(direction.T @ direction - np.eye(3))))
    if gram_error <= DIRECTION_GRAM_TOLERANCE:
        return affine, {
            "direction_repaired": False,
            "direction_gram_error": gram_error,
            "maximum_corner_displacement_mm": 0.0,
        }
    orthogonal, _ = polar(direction)
    if np.sign(np.linalg.det(orthogonal)) != np.sign(np.linalg.det(direction)):
        raise RuntimeError("Direction repair would change handedness")
    corrected = affine.copy()
    corrected[:3, :3] = orthogonal * spacing
    center = (np.asarray(source.shape[:3], dtype=np.float64) - 1.0) / 2.0
    center_world = affine[:3, :3] @ center + affine[:3, 3]
    corrected[:3, 3] = center_world - corrected[:3, :3] @ center
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (0.0, source.shape[0] - 1.0)
            for y in (0.0, source.shape[1] - 1.0)
            for z in (0.0, source.shape[2] - 1.0)
        ],
        dtype=np.float64,
    )
    original_world = (affine @ corners.T).T[:, :3]
    corrected_world = (corrected @ corners.T).T[:, :3]
    maximum_displacement = float(
        np.max(np.linalg.norm(original_world - corrected_world, axis=1))
    )
    relative_limit = float(
        MAXIMUM_CORNER_DISPLACEMENT_FRACTION_OF_MINIMUM_SPACING * np.min(spacing)
    )
    if maximum_displacement > MAXIMUM_CORNER_DISPLACEMENT_MM or maximum_displacement > relative_limit:
        raise RuntimeError(
            "Direction repair exceeds frozen displacement limits: "
            f"{maximum_displacement:.9g} mm, relative limit {relative_limit:.9g} mm"
        )
    if nib.aff2axcodes(corrected) != nib.aff2axcodes(affine):
        raise RuntimeError("Direction repair would change axis codes")
    corrected_direction = corrected[:3, :3] / np.linalg.norm(corrected[:3, :3], axis=0)
    corrected_error = float(
        np.max(np.abs(corrected_direction.T @ corrected_direction - np.eye(3)))
    )
    if corrected_error > 1e-12:
        raise RuntimeError("Polar direction correction is not orthonormal")
    return corrected, {
        "direction_repaired": True,
        "direction_gram_error": gram_error,
        "corrected_direction_gram_error": corrected_error,
        "maximum_corner_displacement_mm": maximum_displacement,
        "relative_displacement_limit_mm": relative_limit,
        "axis_codes": "".join(nib.aff2axcodes(affine)),
        "corrected_axis_codes": "".join(nib.aff2axcodes(corrected)),
        "original_direction_determinant": float(np.linalg.det(direction)),
        "corrected_direction_determinant": float(np.linalg.det(orthogonal)),
    }


def _maximum_corner_displacement(
    first: np.ndarray, second: np.ndarray, shape: tuple[int, ...]
) -> float:
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (0.0, shape[0] - 1.0)
            for y in (0.0, shape[1] - 1.0)
            for z in (0.0, shape[2] - 1.0)
        ],
        dtype=np.float64,
    )
    first_world = (np.asarray(first, dtype=np.float64) @ corners.T).T[:, :3]
    second_world = (np.asarray(second, dtype=np.float64) @ corners.T).T[:, :3]
    return float(np.max(np.linalg.norm(first_world - second_world, axis=1)))


def _mnms_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata_record = common.validated_file(
        common.MNMS_METADATA, cohort="mnms", purpose="read_MnMs_ED_ES_vendor_metadata"
    )
    metadata = pd.read_csv(common.MNMS_METADATA)
    required = {"External code", "ED", "ES", "Vendor", "VendorName", "Pathology"}
    if not required.issubset(metadata.columns):
        raise RuntimeError(f"M&Ms metadata missing columns: {sorted(required - set(metadata.columns))}")
    metadata["External code"] = metadata["External code"].astype(str).str.strip()
    if metadata["External code"].duplicated().any():
        raise RuntimeError("M&Ms metadata contains duplicate External code values")
    for subset in MNMS_SUBSETS:
        common.validated_directory(
            common.MNMS_ROOT / subset,
            cohort="mnms",
            purpose="locate_declared_MnMs_patient_directories",
        )

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = [metadata_record]
    labeled_codes: list[str] = []
    maximum_scaled_label_deviation = 0.0
    canonicalized_phase_count = 0
    direction_repair_records: list[dict[str, Any]] = []
    maximum_serialization_corner_displacement_mm = 0.0
    for _, metadata_row in metadata.iterrows():
        code = str(metadata_row["External code"])
        matches: list[tuple[Path, Path]] = []
        for subset in MNMS_SUBSETS:
            directory = common.MNMS_ROOT / subset / code
            image_path = directory / f"{code}_sa.nii.gz"
            label_path = directory / f"{code}_sa_gt.nii.gz"
            if image_path.is_file() and label_path.is_file():
                matches.append((image_path, label_path))
        if not matches:
            continue
        if len(matches) != 1:
            raise RuntimeError(f"M&Ms patient appears in multiple subsets: {code}")
        image_path, label_path = matches[0]
        image_record = common.validated_file(
            image_path, cohort="mnms", purpose="extract_metadata_declared_ED_ES_images"
        )
        label_record = common.validated_file(
            label_path, cohort="mnms", purpose="validate_and_remap_ED_ES_references"
        )
        records.extend((image_record, label_record))
        image_nii = nib.load(str(image_path.resolve()))
        label_nii = nib.load(str(label_path.resolve()))
        if image_nii.ndim != 4 or label_nii.ndim != 4 or image_nii.shape != label_nii.shape:
            raise RuntimeError(f"M&Ms image/label 4D shape mismatch: {code}")
        if not np.allclose(image_nii.affine, label_nii.affine, rtol=0, atol=1e-5):
            raise RuntimeError(f"M&Ms image/label affine mismatch: {code}")
        if not np.allclose(
            image_nii.header.get_zooms()[:3], label_nii.header.get_zooms()[:3], rtol=0, atol=1e-6
        ):
            raise RuntimeError(f"M&Ms image/label spacing mismatch: {code}")
        corrected_image_affine, image_direction_audit = _bounded_orthonormal_affine(image_nii)
        corrected_label_affine, label_direction_audit = _bounded_orthonormal_affine(label_nii)
        if not np.allclose(corrected_image_affine, corrected_label_affine, rtol=0, atol=1e-9):
            raise RuntimeError(f"M&Ms image/label corrected affine mismatch: {code}")
        if image_direction_audit != label_direction_audit:
            raise RuntimeError(f"M&Ms image/label direction audit mismatch: {code}")
        if image_direction_audit["direction_repaired"]:
            direction_repair_records.append({"patient_id": code, **image_direction_audit})
        ed = _strict_frame(metadata_row["ED"], code=code, name="ED", n_frames=image_nii.shape[3])
        es = _strict_frame(metadata_row["ES"], code=code, name="ES", n_frames=image_nii.shape[3])
        if ed == es:
            raise RuntimeError(f"M&Ms ED and ES are identical: {code}")
        safe_code = re.sub(r"[^A-Za-z0-9]+", "_", code).strip("_")
        if not safe_code:
            raise RuntimeError(f"Cannot form safe identifier for M&Ms code: {code!r}")
        for phase, frame in (("ED", ed), ("ES", es)):
            image = np.asarray(image_nii.dataobj[..., frame], dtype=np.float32)
            raw_float = np.asanyarray(label_nii.dataobj[..., frame])
            try:
                raw, deviation = canonicalize_scaled_labels(raw_float)
            except ValueError as exc:
                raise RuntimeError(f"M&Ms scaled-label gate failed: {code} {phase}: {exc}") from exc
            maximum_scaled_label_deviation = max(maximum_scaled_label_deviation, deviation)
            canonicalized_phase_count += int(deviation > 0.0)
            if not all(np.any(raw == label) for label in (1, 2, 3)):
                raise RuntimeError(f"M&Ms annotated phase lacks a foreground class: {code} {phase}")
            remapped = remap_mnms_to_instrument(raw)
            if not np.array_equal(remap_mnms_to_instrument(remapped), raw):
                raise RuntimeError(f"M&Ms label remap roundtrip failed: {code} {phase}")
            identifier = f"MNMS_{safe_code}_{phase}"
            staged_image = STAGING_DIR / "mnms" / "images" / f"{identifier}_0000.nii.gz"
            staged_reference = STAGING_DIR / "mnms" / "references" / f"{identifier}.nii.gz"
            _save_extracted_nifti(
                image_nii, image, staged_image, np.float32, affine=corrected_image_affine
            )
            _save_extracted_nifti(
                label_nii, remapped, staged_reference, np.uint8, affine=corrected_label_affine
            )
            if not geometry_matches(_sitk_read(staged_image), _sitk_read(staged_reference)):
                raise RuntimeError(f"M&Ms staged image/reference geometry mismatch: {code} {phase}")
            staged_nii = nib.load(str(staged_image.resolve()))
            staged_reference_nii = nib.load(str(staged_reference.resolve()))
            expected_persisted_affine = corrected_image_affine.astype(np.float32).astype(
                np.float64
            )
            if (
                staged_nii.shape != image_nii.shape[:3]
                or staged_reference_nii.shape != label_nii.shape[:3]
                or not np.array_equal(staged_nii.affine, expected_persisted_affine)
                or not np.array_equal(staged_reference_nii.affine, expected_persisted_affine)
            ):
                raise RuntimeError(f"M&Ms extraction changed native geometry: {code} {phase}")
            serialization_displacement = _maximum_corner_displacement(
                corrected_image_affine,
                staged_nii.affine,
                tuple(int(value) for value in image_nii.shape[:3]),
            )
            if serialization_displacement > MAXIMUM_FLOAT32_SERIALIZATION_CORNER_DISPLACEMENT_MM:
                raise RuntimeError(
                    f"M&Ms NIfTI float32 serialization displacement exceeded the frozen limit: "
                    f"{code} {phase} {serialization_displacement:.12g} mm"
                )
            maximum_serialization_corner_displacement_mm = max(
                maximum_serialization_corner_displacement_mm, serialization_displacement
            )
            if nib.aff2axcodes(staged_nii.affine) != nib.aff2axcodes(corrected_image_affine):
                raise RuntimeError(f"M&Ms NIfTI serialization changed axis codes: {code} {phase}")
            rows.append(
                {
                    "cohort": "MnMs",
                    "patient_id": code,
                    "patient_number": np.nan,
                    "vendor": str(metadata_row["Vendor"]),
                    "vendor_name": str(metadata_row["VendorName"]),
                    "pathology": str(metadata_row["Pathology"]),
                    "phase": phase,
                    "frame_index": frame,
                    "identifier": identifier,
                    "image_path": common.relative_path(staged_image),
                    "original_reference_path": common.relative_path(staged_reference),
                    "refined_reference_path": "",
                }
            )
        labeled_codes.append(code)

    frame = pd.DataFrame(rows)
    gate_checks = {
        "expected_labeled_patient_count": len(set(labeled_codes)) == 320,
        "case_count": len(rows) == 640,
        "unique_identifiers": frame["identifier"].nunique() == len(frame),
        "complete_phase_pairs": all(
            set(group["phase"]) == {"ED", "ES"} for _, group in frame.groupby("patient_id")
        ),
        "label_remap_roundtrip": True,
        "native_geometry_preserved": True,
        "metadata_ED_ES_only": True,
        "scaled_label_canonicalization_within_1e_6": maximum_scaled_label_deviation <= 1e-6,
        "bounded_direction_repairs_passed": all(
            record["maximum_corner_displacement_mm"] <= MAXIMUM_CORNER_DISPLACEMENT_MM
            for record in direction_repair_records
        ),
        "exact_expected_float32_affine_serialization": True,
        "float32_serialization_corner_displacement_within_0_001_mm": (
            maximum_serialization_corner_displacement_mm
            <= MAXIMUM_FLOAT32_SERIALIZATION_CORNER_DISPLACEMENT_MM
        ),
    }
    if not all(gate_checks.values()):
        raise RuntimeError(f"M&Ms restricted-cohort gate failed: {gate_checks}")
    checks = {
        **gate_checks,
        "maximum_scaled_label_deviation": maximum_scaled_label_deviation,
        "canonicalized_phase_count": canonicalized_phase_count,
        "direction_repair_patient_count": len(direction_repair_records),
        "direction_repair_records": direction_repair_records,
        "maximum_float32_serialization_corner_displacement_mm": (
            maximum_serialization_corner_displacement_mm
        ),
    }
    return rows, records, checks


def _native_measurement_golden_gate() -> dict[str, Any]:
    mask = np.zeros((2, 4, 4), dtype=np.uint8)
    mask[:, :2, :2] = 1
    volume = native_volume_ml(mask, (1.0, 1.0, 1.0), label=1)
    endpoints = measurement_from_phase_volumes(100.0, 40.0)
    checks = {
        "synthetic_volume_ml": volume == 0.008,
        "synthetic_stroke_volume_ml": endpoints["stroke_volume_ml"] == 60.0,
        "synthetic_rvef_points": endpoints["rvef_points"] == 60.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Native measurement golden gate failed: {checks}")
    return checks


def prepare_restricted_inputs() -> dict[str, Any]:
    common.load_protocol()
    if CASE_MANIFEST_PATH.exists() or common.INPUT_MANIFEST_PATH.exists() or GATE_PATH.exists():
        raise RuntimeError("Restricted input artifacts already exist; automatic replacement is forbidden")
    common.write_status("validating_restricted_inputs", started_at_utc=common.now_utc())
    golden = _native_measurement_golden_gate()
    acdc_rows, acdc_records, acdc_checks = _acdc_rows()
    mnms_rows, mnms_records, mnms_checks = _mnms_rows()
    cases = pd.DataFrame(acdc_rows + mnms_rows).sort_values(
        ["cohort", "patient_id", "phase"]
    )
    if cases["identifier"].duplicated().any():
        raise RuntimeError("Cross-cohort case identifiers are not unique")
    CASE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(CASE_MANIFEST_PATH, index=False)
    gate = {
        "schema_version": 1,
        "decision": "RESTRICTED_COHORT_VALIDATION_GATE_PASS",
        "generated_at_utc": common.now_utc(),
        "protocol_lock": common.file_record(common.LOCK_PATH),
        "native_measurement_golden_gate": golden,
        "acdc": acdc_checks,
        "mnms": mnms_checks,
        "no_MnMs2_access": True,
    }
    common.write_json(GATE_PATH, gate)
    manifest = {
        "schema_version": 1,
        "decision": "RESTRICTED_INPUTS_VALIDATED_AND_STAGED",
        "generated_at_utc": common.now_utc(),
        "case_manifest": common.file_record(CASE_MANIFEST_PATH),
        "validation_gate": common.file_record(GATE_PATH),
        "restricted_input_files": acdc_records + mnms_records,
        "counts": {
            "acdc_patients": 50,
            "acdc_cases": 100,
            "mnms_patients": 320,
            "mnms_cases": 640,
        },
    }
    common.write_json(common.INPUT_MANIFEST_PATH, manifest)
    common.write_status(
        "restricted_inputs_validated",
        input_manifest=common.file_record(common.INPUT_MANIFEST_PATH),
    )
    return manifest


def load_case_manifest() -> pd.DataFrame:
    common.load_protocol()
    gate = common.read_json(GATE_PATH)
    if gate.get("decision") != "RESTRICTED_COHORT_VALIDATION_GATE_PASS":
        raise RuntimeError("Restricted cohort gate did not pass")
    manifest = common.read_json(common.INPUT_MANIFEST_PATH)
    if common.file_record(CASE_MANIFEST_PATH)["sha256"] != manifest["case_manifest"]["sha256"]:
        raise RuntimeError("Case manifest changed after restricted input validation")
    return pd.read_csv(CASE_MANIFEST_PATH, keep_default_na=False)
