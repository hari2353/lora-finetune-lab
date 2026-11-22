from finetune_lab.zero import zero_memory


def test_stage_zero_keeps_everything_replicated():
    out = zero_memory(params=1000, trainable_params=100, num_gpus=4, stage=0)
    assert out["per_gpu_bytes"] == 1000 * 4 + 100 * 4 + 100 * 2 * 4


def test_sharding_reduces_per_gpu_memory():
    base = zero_memory(params=1000, trainable_params=100, num_gpus=4, stage=0)
    s1 = zero_memory(params=1000, trainable_params=100, num_gpus=4, stage=1)
    s2 = zero_memory(params=1000, trainable_params=100, num_gpus=4, stage=2)
    s3 = zero_memory(params=1000, trainable_params=100, num_gpus=4, stage=3)
    assert base["per_gpu_bytes"] > s1["per_gpu_bytes"] > s2["per_gpu_bytes"] > s3["per_gpu_bytes"]


def test_single_gpu_all_stages_equal():
    for stage in (0, 1, 2, 3):
        out = zero_memory(params=500, trainable_params=50, num_gpus=1, stage=stage)
        assert out["per_gpu_bytes"] == 500 * 4 + 50 * 4 + 50 * 2 * 4


def test_stage_three_shards_parameters():
    out = zero_memory(params=1000, trainable_params=100, num_gpus=4, stage=3)
    assert out["per_gpu_bytes"] == (1000 * 4 + 100 * 4 + 100 * 2 * 4) / 4


def test_invalid_inputs_rejected():
    cases = [
        {"stage": 4},
        {"num_gpus": 0},
        {"params": -1},
        {"optimizer_states_per_param": -1},
        {"bytes_per_param": 0},
        {"params": 1000, "trainable_params": 2000},
        {"params": 1000, "trainable_params": -1},
    ]
    for kwargs in cases:
        base = {"params": 1000, "trainable_params": 100}
        base.update(kwargs)
        try:
            zero_memory(**base)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")


def test_default_trainable_params_equals_params():
    out = zero_memory(params=100, num_gpus=2, stage=0)
    assert out["trainable_params"] == 100
