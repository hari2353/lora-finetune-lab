"""DeepSpeed ZeRO memory planner (cross-pollinated from deepspeedai/DeepSpeed).

Estimates per-GPU memory for a training run at each ZeRO stage, so the
stage choice can be made from numbers instead of guesswork. Pure arithmetic;
no DeepSpeed dependency.
"""

from __future__ import annotations

_VALID_STAGES = {0, 1, 2, 3}


def zero_memory(
    *,
    params: int,
    trainable_params: int | None = None,
    optimizer_states_per_param: int = 2,
    num_gpus: int = 1,
    stage: int = 0,
    bytes_per_param: float = 4.0,
    bytes_per_grad: float = 4.0,
    bytes_per_opt_state: float = 4.0,
) -> dict:
    """Per-GPU memory estimate for ZeRO stages 0-3.

    Stage 0 keeps the full model, gradients and optimizer states on every
    GPU. Stage 1 shards optimizer states, stage 2 also shards gradients,
    stage 3 also shards parameters.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(_VALID_STAGES)}")
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    if params < 0:
        raise ValueError("params must be >= 0")
    if optimizer_states_per_param < 0:
        raise ValueError("optimizer_states_per_param must be >= 0")
    for label, value in (
        ("bytes_per_param", bytes_per_param),
        ("bytes_per_grad", bytes_per_grad),
        ("bytes_per_opt_state", bytes_per_opt_state),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be > 0")
    trainable = params if trainable_params is None else trainable_params
    if trainable < 0 or trainable > params:
        raise ValueError("trainable_params must be between 0 and params")

    model_bytes = params * bytes_per_param
    grad_bytes = trainable * bytes_per_grad
    opt_bytes = trainable * optimizer_states_per_param * bytes_per_opt_state

    if stage == 0:
        per_gpu = model_bytes + grad_bytes + opt_bytes
    elif stage == 1:
        per_gpu = model_bytes + grad_bytes + opt_bytes / num_gpus
    elif stage == 2:
        per_gpu = model_bytes + grad_bytes / num_gpus + opt_bytes / num_gpus
    else:
        per_gpu = model_bytes / num_gpus + grad_bytes / num_gpus + opt_bytes / num_gpus

    total = per_gpu * num_gpus
    return {
        "stage": stage,
        "num_gpus": num_gpus,
        "params": params,
        "trainable_params": trainable,
        "model_bytes": round(model_bytes, 2),
        "gradient_bytes": round(grad_bytes, 2),
        "optimizer_bytes": round(opt_bytes, 2),
        "per_gpu_bytes": round(per_gpu, 2),
        "total_bytes": round(total, 2),
    }
