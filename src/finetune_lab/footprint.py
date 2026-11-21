"""Training footprint and LoRA parameter accounting.

Cross-pollinated from harvard-edge/TinyML-Footprint: report operational
energy and carbon for a fine-tune instead of only F1. GPU-hours is the
measured quantity; power and carbon intensity are explicit assumptions.
"""

from __future__ import annotations

JOULES_PER_KWH = 3_600_000.0


def training_energy(
    gpu_hours: float,
    gpu_power_w: float,
    *,
    gpus: int = 1,
    pue: float = 1.15,
) -> dict:
    """Operational energy for a training run (kWh)."""
    if gpu_hours <= 0 or gpu_power_w <= 0:
        raise ValueError("gpu_hours and gpu_power_w must be positive")
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    if pue < 1.0:
        raise ValueError("pue must be >= 1.0")
    kwh = gpu_hours * gpus * gpu_power_w * pue / 1000
    return {
        "gpu_hours": round(gpu_hours, 4),
        "gpus": gpus,
        "kwh": round(kwh, 4),
    }


def training_carbon(
    gpu_hours: float,
    gpu_power_w: float,
    *,
    carbon_intensity_g_per_kwh: float = 400.0,
    gpus: int = 1,
    pue: float = 1.15,
) -> dict:
    """Operational kgCO2e for a training run."""
    if carbon_intensity_g_per_kwh < 0:
        raise ValueError("carbon intensity must be >= 0")
    energy = training_energy(gpu_hours, gpu_power_w, gpus=gpus, pue=pue)
    kg = energy["kwh"] * carbon_intensity_g_per_kwh / 1000
    return {
        **energy,
        "carbon_intensity_g_per_kwh": carbon_intensity_g_per_kwh,
        "kgco2e": round(kg, 4),
    }


def lora_trainable_params(rank: int, modules: list[tuple[int, int]]) -> dict:
    """Trainable parameters for LoRA on ``modules`` = [(in_features, out_features)].

    Each adapted matrix gets A (r x in) and B (out x r), so r*(in+out)
    trainable values per module. This is the number that goes in a model
    datasheet, not the base-model parameter count."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if not modules:
        raise ValueError("modules must be non-empty")
    for in_features, out_features in modules:
        if in_features < 1 or out_features < 1:
            raise ValueError("module dimensions must be positive")
    total = rank * sum(i + o for i, o in modules)
    return {"rank": rank, "modules": len(modules), "trainable_params": total}


def lora_share(trainable_params: int, base_params: int) -> float:
    """Fraction of the full model that training actually updates."""
    if trainable_params < 0 or base_params <= 0:
        raise ValueError("invalid parameter counts")
    return round(trainable_params / base_params, 6)
