import json
from pathlib import Path

import pytest

from finetune_lab.data import generate_pairs, load_csv, save_csv, to_instruction_format
from finetune_lab.mock_trainer import MultiLabelTrainer, micro_f1
from finetune_lab.train_lora import DEFAULT_LORA_CONFIG, build_training_script, peft_available

LABELS = ["esco__001", "esco__004", "esco__005", "esco__007", "esco__011", "esco__016"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    pairs = generate_pairs(n_per_skill=25, seed=42)
    save_csv(pairs, tmp / "pairs.csv")
    return load_csv(tmp / "pairs.csv")


def test_generation_deterministic_and_labeled(dataset):
    again = generate_pairs(n_per_skill=25, seed=42)
    assert [p["text"] for p in dataset] == [p["text"] for p in again]
    assert all(p["labels"] for p in dataset)
    assert len(dataset) == 6 * 25


def test_instruction_format_is_valid_json_assistant(dataset):
    names = {sid: sid for sid in LABELS}
    records = to_instruction_format(dataset[:5], names)
    for record in records:
        assistant = record["messages"][-1]["content"]
        parsed = json.loads(assistant)
        assert set(parsed) == {"skills"}


def test_trained_model_beats_random_baseline(dataset):
    rng_labels = LABELS
    random_preds = [{rng_labels[i % len(rng_labels)]} for i in range(len(dataset))]
    truth = [set(p["labels"]) for p in dataset]
    baseline = micro_f1(truth, random_preds)

    split = int(len(dataset) * 0.8)
    train, test = dataset[:split], dataset[split:]
    trainer = MultiLabelTrainer(label_space=LABELS)
    stats = trainer.fit(train)
    assert stats["final_loss"] < stats["initial_loss"]

    preds = [set(trainer.predict(p["text"])) for p in test]
    tuned = micro_f1([set(p["labels"]) for p in test], preds)
    assert tuned["f1"] > baseline["f1"]
    artifact = Path(__file__).parent / "artifacts_test.json"
    trainer.save(artifact)
    loaded = MultiLabelTrainer.load(artifact)
    assert loaded.predict(test[0]["text"]) == trainer.predict(test[0]["text"])
    artifact.unlink(missing_ok=True)


def test_lora_script_generation_contains_config():
    script = build_training_script("Qwen/Qwen2.5-0.5B-Instruct", "train.jsonl", "out")
    assert str(DEFAULT_LORA_CONFIG["r"]) in script
    assert "Qwen/Qwen2.5-0.5B-Instruct" in script


def test_peft_optional():
    assert isinstance(peft_available(), bool)
