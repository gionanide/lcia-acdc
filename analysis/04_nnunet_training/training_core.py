"""Official nnU-Net training with read-only qualification instrumentation."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    from .training_io import (
        assert_output_path,
        file_record,
        now_utc,
        relative_path,
        set_nnunet_environment,
    )
except ImportError:
    from training_io import (  # type: ignore[no-redef]
        assert_output_path,
        file_record,
        now_utc,
        relative_path,
        set_nnunet_environment,
    )

set_nnunet_environment()

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (  # noqa: E402
    nnUNetTrainer,
)


AUDIT_FIELDS = (
    "stage",
    "completed_epoch",
    "recorded_at_utc",
    "epoch_seconds",
    "stage_elapsed_seconds",
    "learning_rate",
    "train_loss",
    "validation_loss",
    "mean_foreground_dice",
    "ema_foreground_dice",
    "rv_pseudo_dice",
    "myocardium_pseudo_dice",
    "lv_pseudo_dice",
    "network_forward_calls",
    "loss_forward_calls",
    "optimizer_steps_expected",
    "optimizer_steps_executed",
    "optimizer_steps_skipped",
    "grad_scaler_updates",
    "grad_scaler_scale_decreases",
    "grad_scaler_scale",
    "finite_inputs",
    "finite_targets",
    "finite_logits",
    "finite_losses",
    "finite_metrics",
    "finite_model_parameters",
    "finite_model_buffers",
    "finite_gradients",
    "finite_optimizer_state",
    "finite_grad_scaler_state",
    "finite_metric_ema",
    "parameter_ema_present",
    "finite_epoch",
    "peak_cuda_memory_bytes",
)


def _numeric_scalar_finite(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _finite_checks(value: Any) -> list[torch.Tensor | bool]:
    """Return asynchronous scalar checks without synchronizing each GPU tensor."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return [torch.isfinite(value.detach()).all()]
        return [True]
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return [bool(np.isfinite(value).all())]
        return [True]
    if isinstance(value, dict):
        checks: list[torch.Tensor | bool] = []
        for item in value.values():
            checks.extend(_finite_checks(item))
        return checks
    if isinstance(value, (list, tuple)):
        checks = []
        for item in value:
            checks.extend(_finite_checks(item))
        return checks
    return [_numeric_scalar_finite(value)]


def _resolve_checks(checks: Iterable[torch.Tensor | bool]) -> bool:
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    cpu_flags: list[bool] = []
    for check in checks:
        if isinstance(check, torch.Tensor):
            by_device.setdefault(check.device, []).append(check.reshape(()))
        else:
            cpu_flags.append(bool(check))
    if not all(cpu_flags):
        return False
    for values in by_device.values():
        if not bool(torch.stack(values).all().item()):
            return False
    return True


def tree_is_finite(value: Any) -> bool:
    return _resolve_checks(_finite_checks(value))


def _module_state_finite(
    module: torch.nn.Module,
) -> tuple[bool, bool, bool]:
    parameters = list(module.parameters())
    buffers = list(module.buffers())
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    return (
        tree_is_finite(parameters),
        tree_is_finite(buffers),
        tree_is_finite(gradients),
    )


def _append_epoch_audit(path: Path, row: dict[str, Any]) -> None:
    destination = assert_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.is_file()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row[field] for field in AUDIT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


class QualificationAuditMixin:
    """Instrumentation only; the official optimization steps remain inherited."""

    audit_path: Path | None = None
    audit_stage: str | None = None
    audit_stage_started: float | None = None

    def configure_qualification_audit(
        self,
        *,
        path: Path,
        stage: str,
        stage_started: float,
    ) -> None:
        destination = assert_output_path(path)
        if destination.exists():
            raise RuntimeError(f"Epoch audit already exists: {destination}")
        if stage_started <= 0:
            raise ValueError("Invalid audit timing configuration")
        self.audit_path = destination
        self.audit_stage = stage
        self.audit_stage_started = float(stage_started)

    def _network_input_hook(
        self, _module: torch.nn.Module, args: tuple[Any, ...]
    ) -> None:
        self._epoch_network_forward_calls += 1
        self._epoch_input_checks.extend(_finite_checks(args))

    def _network_output_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        self._epoch_logit_checks.extend(_finite_checks(output))

    def _loss_input_hook(self, _module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        self._epoch_loss_forward_calls += 1
        if len(args) >= 2:
            self._epoch_target_checks.extend(_finite_checks(args[1]))
        else:
            self._epoch_target_checks.append(False)

    def _optimizer_step_post_hook(
        self,
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self._epoch_optimizer_steps += 1

    def _install_instrumentation(self) -> None:
        self._network_pre_handle = self.network.register_forward_pre_hook(
            self._network_input_hook
        )
        self._network_post_handle = self.network.register_forward_hook(
            self._network_output_hook
        )
        self._loss_pre_handle = self.loss.register_forward_pre_hook(
            self._loss_input_hook
        )
        self._optimizer_post_handle = self.optimizer.register_step_post_hook(
            self._optimizer_step_post_hook
        )
        self._original_scaler_update = None
        if self.grad_scaler is not None:
            original_update = self.grad_scaler.update
            self._original_scaler_update = original_update

            def audited_update(new_scale: Any = None) -> Any:
                before = float(self.grad_scaler.get_scale())
                if new_scale is None:
                    result = original_update()
                else:
                    result = original_update(new_scale)
                after = float(self.grad_scaler.get_scale())
                self._epoch_scaler_updates += 1
                if after < before:
                    self._epoch_scaler_scale_decreases += 1
                return result

            self.grad_scaler.update = audited_update

    def _remove_instrumentation(self) -> None:
        for name in (
            "_network_pre_handle",
            "_network_post_handle",
            "_loss_pre_handle",
            "_optimizer_post_handle",
        ):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.remove()
                setattr(self, name, None)
        if self.grad_scaler is not None and self._original_scaler_update is not None:
            self.grad_scaler.update = self._original_scaler_update
            self._original_scaler_update = None

    def _reset_epoch_audit(self) -> None:
        self._epoch_input_checks: list[torch.Tensor | bool] = []
        self._epoch_target_checks: list[torch.Tensor | bool] = []
        self._epoch_logit_checks: list[torch.Tensor | bool] = []
        self._epoch_network_forward_calls = 0
        self._epoch_loss_forward_calls = 0
        self._epoch_optimizer_steps = 0
        self._epoch_scaler_updates = 0
        self._epoch_scaler_scale_decreases = 0

    def on_train_start(self) -> None:
        super().on_train_start()
        self._reset_epoch_audit()
        self._install_instrumentation()

    def on_epoch_start(self) -> None:
        self._reset_epoch_audit()
        super().on_epoch_start()

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch + 1
        super().on_epoch_end()
        logging = self.logger
        train_loss = float(logging.get_value("train_losses", step=-1))
        validation_loss = float(logging.get_value("val_losses", step=-1))
        learning_rate = float(logging.get_value("lrs", step=-1))
        mean_dice = float(logging.get_value("mean_fg_dice", step=-1))
        ema_dice = float(logging.get_value("ema_fg_dice", step=-1))
        class_dice = [
            float(value)
            for value in logging.get_value("dice_per_class_or_region", step=-1)
        ]
        if len(class_dice) != 3:
            raise RuntimeError("Official foreground pseudo-Dice contract changed")
        epoch_seconds = float(
            logging.get_value("epoch_end_timestamps", step=-1)
            - logging.get_value("epoch_start_timestamps", step=-1)
        )
        parameters_finite, buffers_finite, gradients_finite = _module_state_finite(
            self.network
        )
        scaler_state = (
            self.grad_scaler.state_dict() if self.grad_scaler is not None else None
        )
        expected_steps = int(self.num_iterations_per_epoch)
        skipped_steps = expected_steps - int(self._epoch_optimizer_steps)
        stage_elapsed = time.perf_counter() - float(self.audit_stage_started)
        finite_metrics = tree_is_finite(
            [learning_rate, mean_dice, ema_dice, epoch_seconds, *class_dice]
        )
        finite_losses = tree_is_finite([train_loss, validation_loss])
        finite_inputs = _resolve_checks(self._epoch_input_checks)
        finite_targets = _resolve_checks(self._epoch_target_checks)
        finite_logits = _resolve_checks(self._epoch_logit_checks)
        finite_optimizer_state = tree_is_finite(self.optimizer.state_dict())
        finite_scaler_state = tree_is_finite(scaler_state)
        finite_metric_ema = tree_is_finite([ema_dice, self._best_ema])
        expected_forward_calls = int(
            self.num_iterations_per_epoch + self.num_val_iterations_per_epoch
        )
        finite_epoch = all(
            (
                self._epoch_network_forward_calls == expected_forward_calls,
                self._epoch_loss_forward_calls == expected_forward_calls,
                finite_inputs,
                finite_targets,
                finite_logits,
                finite_losses,
                finite_metrics,
                parameters_finite,
                buffers_finite,
                finite_optimizer_state,
                finite_scaler_state,
                finite_metric_ema,
                0 <= skipped_steps <= expected_steps,
                (
                    self._epoch_scaler_updates == expected_steps
                    if self.grad_scaler is not None
                    else skipped_steps == 0
                ),
            )
        )
        row = {
            "stage": self.audit_stage,
            "completed_epoch": completed_epoch,
            "recorded_at_utc": now_utc(),
            "epoch_seconds": epoch_seconds,
            "stage_elapsed_seconds": stage_elapsed,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "mean_foreground_dice": mean_dice,
            "ema_foreground_dice": ema_dice,
            "rv_pseudo_dice": class_dice[0],
            "myocardium_pseudo_dice": class_dice[1],
            "lv_pseudo_dice": class_dice[2],
            "network_forward_calls": self._epoch_network_forward_calls,
            "loss_forward_calls": self._epoch_loss_forward_calls,
            "optimizer_steps_expected": expected_steps,
            "optimizer_steps_executed": self._epoch_optimizer_steps,
            "optimizer_steps_skipped": skipped_steps,
            "grad_scaler_updates": self._epoch_scaler_updates,
            "grad_scaler_scale_decreases": self._epoch_scaler_scale_decreases,
            "grad_scaler_scale": (
                float(self.grad_scaler.get_scale())
                if self.grad_scaler is not None
                else ""
            ),
            "finite_inputs": finite_inputs,
            "finite_targets": finite_targets,
            "finite_logits": finite_logits,
            "finite_losses": finite_losses,
            "finite_metrics": finite_metrics,
            "finite_model_parameters": parameters_finite,
            "finite_model_buffers": buffers_finite,
            "finite_gradients": gradients_finite,
            "finite_optimizer_state": finite_optimizer_state,
            "finite_grad_scaler_state": finite_scaler_state,
            "finite_metric_ema": finite_metric_ema,
            "parameter_ema_present": False,
            "finite_epoch": finite_epoch,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
        }
        if self.audit_path is None:
            raise RuntimeError("Qualification audit was not configured")
        _append_epoch_audit(self.audit_path, row)
        if not finite_epoch:
            raise FloatingPointError(
                f"Nonfinite or incomplete official epoch audit at epoch {completed_epoch}"
            )

    def on_train_end(self) -> None:
        try:
            super().on_train_end()
        finally:
            self._remove_instrumentation()


class QualificationNNUNetTrainer(QualificationAuditMixin, nnUNetTrainer):
    """Official v2.8.1 trainer plus qualification instrumentation."""


class QualificationNNUNetContinuationTrainer(QualificationAuditMixin, nnUNetTrainer):
    """Nonselectable fold-0 continuation diagnostic."""


def close_training_resources(trainer: nnUNetTrainer | None) -> None:
    if trainer is None:
        return
    if isinstance(trainer, QualificationAuditMixin):
        trainer._remove_instrumentation()
    for name in ("dataloader_train", "dataloader_val"):
        loader = getattr(trainer, name, None)
        finish = getattr(loader, "_finish", None)
        if callable(finish):
            try:
                finish()
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def audit_checkpoint(
    path: Path,
    *,
    expected_epoch: int | None = None,
    selectable: bool,
) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise RuntimeError(f"Expected checkpoint is missing: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = {
        "network_weights",
        "optimizer_state",
        "grad_scaler_state",
        "logging",
        "_best_ema",
        "current_epoch",
        "init_args",
        "trainer_name",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint is missing fields: {missing}")
    epoch = int(checkpoint["current_epoch"])
    if expected_epoch is not None and epoch != expected_epoch:
        raise RuntimeError(f"Checkpoint epoch mismatch: {epoch} != {expected_epoch}")
    logging = checkpoint["logging"]
    expected_logging = {
        "mean_fg_dice",
        "ema_fg_dice",
        "dice_per_class_or_region",
        "train_losses",
        "val_losses",
        "lrs",
        "epoch_start_timestamps",
        "epoch_end_timestamps",
    }
    if set(logging) != expected_logging:
        raise RuntimeError("Unexpected official logger checkpoint contract")
    lengths = {name: len(values) for name, values in logging.items()}
    if set(lengths.values()) != {epoch}:
        raise RuntimeError(
            f"Checkpoint logger lengths do not match epoch {epoch}: {lengths}"
        )
    finite_components = {
        "network_weights": tree_is_finite(checkpoint["network_weights"]),
        "optimizer_state": tree_is_finite(checkpoint["optimizer_state"]),
        "grad_scaler_state": tree_is_finite(checkpoint["grad_scaler_state"]),
        "logging": tree_is_finite(logging),
        "metric_ema": tree_is_finite(checkpoint["_best_ema"]),
    }
    if not all(finite_components.values()):
        raise FloatingPointError(f"Nonfinite checkpoint state: {finite_components}")
    return {
        **file_record(source),
        "current_epoch": epoch,
        "trainer_name": str(checkpoint["trainer_name"]),
        "logging_lengths": lengths,
        "finite_components": finite_components,
        "all_finite": True,
        "production_selectable": bool(selectable),
        "checkpoint_rule": source.name,
    }
