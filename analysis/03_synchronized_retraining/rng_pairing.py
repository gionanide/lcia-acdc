#!/usr/bin/env python
"""Torch RNG capture/restore helpers and synchronized paired update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from common import hash_rng_state


def capture_torch_rng() -> Dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_torch_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if not torch.equal(a["cpu"], b["cpu"]):
        return False
    if len(a["cuda"]) != len(b["cuda"]):
        return False
    return all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))


def optimizer_state_max_abs_diff(a: torch.optim.Optimizer, b: torch.optim.Optimizer) -> float:
    max_diff = 0.0
    a_state = a.state_dict()["state"]
    b_state = b.state_dict()["state"]
    for key in set(a_state) | set(b_state):
        av = a_state.get(key, {})
        bv = b_state.get(key, {})
        for sk in set(av) | set(bv):
            if sk not in av or sk not in bv:
                return float("inf")
            x, y = av[sk], bv[sk]
            if torch.is_tensor(x) and torch.is_tensor(y):
                if x.numel():
                    max_diff = max(max_diff, float((x - y).abs().max().item()))
            elif x != y:
                return float("inf")
    return max_diff


def model_parameter_max_abs_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    max_diff = 0.0
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.numel():
            max_diff = max(max_diff, float((pa.detach() - pb.detach()).abs().max().item()))
    for (_, ba), (_, bb) in zip(a.named_buffers(), b.named_buffers()):
        if torch.is_floating_point(ba) and ba.numel():
            max_diff = max(max_diff, float((ba.detach() - bb.detach()).abs().max().item()))
        elif not torch.equal(ba, bb):
            return float("inf")
    return max_diff


def gradient_max_abs_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    max_diff = 0.0
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None and pb.grad is None:
            continue
        if pa.grad is None or pb.grad is None:
            return float("inf")
        if pa.grad.numel():
            max_diff = max(max_diff, float((pa.grad.detach() - pb.grad.detach()).abs().max().item()))
    return max_diff


@dataclass
class PairedStepResult:
    loss_a: float
    loss_b: float
    max_logit_diff: float
    max_gradient_diff: float
    max_parameter_diff_after: float
    max_optimizer_state_diff_after: float
    rng_before_hash: Dict[str, Any]
    rng_after_a_hash: Dict[str, Any]
    rng_after_b_hash: Dict[str, Any]
    expected_after_equal: bool


def paired_train_step(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    optimizer_a: torch.optim.Optimizer,
    optimizer_b: torch.optim.Optimizer,
    loss_fn,
    x: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    *,
    amp: bool = False,
    scaler_a: Optional[torch.amp.GradScaler] = None,
    scaler_b: Optional[torch.amp.GradScaler] = None,
) -> PairedStepResult:
    optimizer_a.zero_grad(set_to_none=True)
    optimizer_b.zero_grad(set_to_none=True)
    state_before = capture_torch_rng()
    hash_before = hash_rng_state()

    with torch.autocast("cuda", enabled=amp and x.is_cuda):
        logits_a = model_a(x)
        loss_a = loss_fn(logits_a, y_a)
    if scaler_a is not None and scaler_a.is_enabled():
        scaler_a.scale(loss_a).backward()
        scaler_a.step(optimizer_a)
        scaler_a.update()
    else:
        loss_a.backward()
        optimizer_a.step()
    state_after_a = capture_torch_rng()
    hash_after_a = hash_rng_state()

    restore_torch_rng(state_before)
    with torch.autocast("cuda", enabled=amp and x.is_cuda):
        logits_b = model_b(x)
        loss_b = loss_fn(logits_b, y_b)
    if scaler_b is not None and scaler_b.is_enabled():
        scaler_b.scale(loss_b).backward()
        scaler_b.step(optimizer_b)
        scaler_b.update()
    else:
        loss_b.backward()
        optimizer_b.step()
    state_after_b = capture_torch_rng()
    hash_after_b = hash_rng_state()

    return PairedStepResult(
        loss_a=float(loss_a.detach().cpu().item()),
        loss_b=float(loss_b.detach().cpu().item()),
        max_logit_diff=float((logits_a.detach() - logits_b.detach()).abs().max().cpu().item()),
        max_gradient_diff=gradient_max_abs_diff(model_a, model_b),
        max_parameter_diff_after=model_parameter_max_abs_diff(model_a, model_b),
        max_optimizer_state_diff_after=optimizer_state_max_abs_diff(optimizer_a, optimizer_b),
        rng_before_hash=hash_before,
        rng_after_a_hash=hash_after_a,
        rng_after_b_hash=hash_after_b,
        expected_after_equal=rng_states_equal(state_after_a, state_after_b),
    )
