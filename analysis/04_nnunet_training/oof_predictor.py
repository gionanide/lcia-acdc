"""Shared official nnU-Net predictor construction for OOF inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from .training_io import (
        PREPROCESSED_DATASET,
        file_record,
        read_json,
        set_nnunet_environment,
    )
except ImportError:
    from training_io import (  # type: ignore[no-redef]
        PREPROCESSED_DATASET,
        file_record,
        read_json,
        set_nnunet_environment,
    )

set_nnunet_environment()

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (  # noqa: E402
    nnUNetTrainer,
)
from nnunetv2.utilities.label_handling.label_handling import (  # noqa: E402
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import (  # noqa: E402
    PlansManager,
)


def build_predictor(checkpoint_path: Path) -> tuple[nnUNetPredictor, dict[str, Any]]:
    plans = read_json(PREPROCESSED_DATASET / "nnUNetPlans.json")
    dataset = read_json(PREPROCESSED_DATASET / "dataset.json")
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint.get("network_weights"), dict):
        raise RuntimeError("Invalid official nnU-Net checkpoint weights")
    configuration_name = str(checkpoint["init_args"]["configuration"])
    if configuration_name != "2d":
        raise RuntimeError("OOF checkpoint is not the frozen 2D configuration")
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    label_manager = plans_manager.get_label_manager(dataset)
    input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset
    )
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
    return predictor, {
        "checkpoint_current_epoch": int(checkpoint["current_epoch"]),
        "checkpoint_trainer_name": str(checkpoint["trainer_name"]),
        "configuration": configuration_name,
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "perform_everything_on_device": True,
        "save_probabilities": False,
        "num_processes_preprocessing": 2,
        "num_processes_segmentation_export": 2,
        "native_grid_export": True,
    }


def prediction_records(
    output_dir: Path, identifiers: list[str]
) -> list[dict[str, Any]]:
    expected = {f"{identifier}.nii.gz" for identifier in identifiers}
    actual = {path.name for path in output_dir.glob("*.nii.gz")}
    if actual != expected:
        raise RuntimeError(
            f"Incomplete native-grid prediction set in {output_dir}: "
            f"expected {len(expected)}, found {len(actual)}"
        )
    return [file_record(output_dir / name) for name in sorted(expected)]
