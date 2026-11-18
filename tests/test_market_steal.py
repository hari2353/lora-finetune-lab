"""Market-steal additions: stratified split + per-class report, honest run
logging, plateau early-stop + cosine schedule — the hygiene every serious
finetune lab (Axolotl, LLaMA-Factory, torchtune) treats as baseline."""

import json

import pytest

from finetune_lab.data import generate_pairs, stratified_split
from finetune_lab.mock_trainer import (
    ExperimentLog,
    MultiLabelTrainer,
    micro_f1,
    per_class_report,
)

LABELS = ["esco__001", "esco__004", "esco__005", "esco__007", "esco__011", "esco__016"]


@pytest.fixture(scope="module")
def dataset():
    return generate_pairs(n_per_skill=25, seed=42)


# ── stratified split + per-class report ──────────────────────────────────────
def test_split_is_stratified_disjoint_deterministic(dataset) -> None:
    first = stratified_split(dataset, seed=42)
    second = stratified_split(dataset, seed=42)
    assert [p["text"] for p in first["train"]] == [p["text"] for p in second["train"]]
    positions = [id(p) for part in first.values() for p in part]
    assert len(positions) == len(set(positions))  # disjoint by position, duplicates included
    assert len(positions) == len(dataset)
    train, val, test = (len(first[k]) for k in ("train", "val", "test"))
    assert abs(train - 0.8 * len(dataset)) <= 6  # proportions honored
    # every skill appears in every part (stratification, not luck)
    for part in first.values():
        seen = {label for p in part for label in p["labels"]}
        assert seen == set(LABELS)


def test_per_class_report_counts_every_label() -> None:
    y_true = [{"a"}, {"a", "b"}, {"c"}, {"c"}]
    y_pred = [{"a"}, {"b"}, {"c"}, {"a"}]
    report = per_class_report(y_true, y_pred, labels=["a", "b", "c"])
    assert set(report) == {"a", "b", "c", "macro_f1"}
    assert report["a"]["support"] == 2 and report["c"]["support"] == 2
    assert report["b"]["precision"] == 1.0 and report["b"]["f1"] > 0
    assert report["c"]["precision"] == 1.0 and report["c"]["recall"] == 0.5
    assert 0.0 <= report["macro_f1"] <= 1.0


# ── honest run log ───────────────────────────────────────────────────────────
def test_run_log_records_config_fingerprint_metrics_and_picks_best(tmp_path, dataset) -> None:
    log = ExperimentLog(tmp_path / "runs.jsonl")
    split = stratified_split(dataset, seed=42)

    for lr in (0.6, 0.3):
        trainer = MultiLabelTrainer(label_space=LABELS, lr=lr, epochs=10)
        stats = trainer.fit(split["train"], val_pairs=split["val"])
        log.append(trainer=trainer, config={"lr": lr, "epochs": 10},
                   dataset=split["train"], stats=stats)

    rows = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["dataset_fingerprint"] == rows[1]["dataset_fingerprint"]  # same data
    assert rows[0]["config"]["lr"] == 0.6
    assert "macro_f1" in rows[0]["metrics"] and "final_loss" in rows[0]["metrics"]
    best = log.best()
    assert best["config"]["lr"] in (0.6, 0.3)
    f1s = [row["metrics"]["macro_f1"] for row in rows]
    assert best["metrics"]["macro_f1"] == max(f1s)


def test_same_seed_runs_are_reproducible(tmp_path, dataset) -> None:
    split = stratified_split(dataset, seed=42)
    fingerprints = []
    histories = []
    for _ in range(2):
        trainer = MultiLabelTrainer(label_space=LABELS, epochs=5)
        stats = trainer.fit(split["train"])
        fingerprints.append(MultiLabelTrainer.dataset_fingerprint(split["train"]))
        histories.append(trainer.history)
    assert fingerprints[0] == fingerprints[1] and histories[0] == histories[1]


# ── early stop + cosine schedule ────────────────────────────────────────────
def test_plateau_triggers_early_stop_only_after_patience(dataset) -> None:
    split = stratified_split(dataset, seed=42)
    trainer = MultiLabelTrainer(label_space=LABELS, epochs=60, lr=0.6)
    stats = trainer.fit(split["train"], val_pairs=split["val"], patience=3)
    assert stats["stopped_early"] is True
    assert stats["epochs_run"] < 60
    # a healthy run must NOT stop early
    healthy = MultiLabelTrainer(label_space=LABELS, epochs=6, lr=0.6)
    healthy_stats = healthy.fit(split["train"], val_pairs=split["val"], patience=3)
    assert healthy_stats["stopped_early"] is False
    assert healthy_stats["epochs_run"] == 6


def test_cosine_schedule_decays_lr(dataset) -> None:
    split = stratified_split(dataset, seed=42)
    trainer = MultiLabelTrainer(label_space=LABELS, epochs=12, lr=0.6)
    stats = trainer.fit(split["train"])
    assert stats["lr_initial"] == pytest.approx(0.6)
    assert stats["lr_final"] < stats["lr_initial"] * 0.9  # cosine decayed
    assert trainer.history_lr[0] > trainer.history_lr[-1]
