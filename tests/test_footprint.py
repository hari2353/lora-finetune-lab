from finetune_lab.footprint import (
    lora_share,
    lora_trainable_params,
    training_carbon,
    training_energy,
)


def test_training_energy_math():
    out = training_energy(2.0, 400, pue=1.0)
    assert out["kwh"] == 0.8


def test_pue_gpus_scale_energy():
    out = training_energy(2.0, 400, pue=1.2, gpus=4)
    assert out["kwh"] == 3.84


def test_training_carbon_math():
    out = training_carbon(2.0, 400, pue=1.0, carbon_intensity_g_per_kwh=400)
    assert out["kgco2e"] == 0.32


def test_energy_rejects_bad_inputs():
    for kwargs in (
        {"gpu_hours": 0, "gpu_power_w": 400},
        {"gpu_hours": 1, "gpu_power_w": 0},
        {"gpu_hours": 1, "gpu_power_w": 400, "gpus": 0},
        {"gpu_hours": 1, "gpu_power_w": 400, "pue": 0.9},
    ):
        try:
            training_energy(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")


def test_lora_trainable_params_math():
    # Qwen2.5-0.5B-ish: q/k/v/o projections, hidden 896
    modules = [(896, 896)] * 4
    out = lora_trainable_params(16, modules)
    assert out["trainable_params"] == 16 * 4 * (896 + 896)
    assert out["modules"] == 4


def test_lora_share_is_small():
    assert lora_share(114_688, 494_032_768) == 0.000232


def test_lora_validation():
    for args in ((0, [(8, 8)]), (8, []), (8, [(0, 8)])):
        try:
            lora_trainable_params(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {args}")
