from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash_vec(tokens: list[str], dim: int = 512) -> np.ndarray:
    import hashlib

    vec = np.zeros(dim, dtype=np.float32)
    grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    for g in grams:
        d = hashlib.md5(g.encode()).digest()
        head = int.from_bytes(d[:8], "little")
        sign = 1.0 if d[8] & 1 else -1.0
        vec[head % dim] += sign
    n = float(np.linalg.norm(vec))
    return vec / n if n else vec


class MultiLabelTrainer:
    """Offline stand-in for a LoRA adapter run: per-skill linear heads over hashed features,
    trained with SGD. Same lifecycle as the real path (fit -> save -> load -> predict),
    including its cosine LR schedule and early-stopping-on-plateau callback."""

    def __init__(self, label_space: list[str], dim: int = 512, lr: float = 0.6, epochs: int = 80) -> None:
        self.label_space = label_space
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.W = np.zeros((len(label_space), dim), dtype=np.float32)
        self.b = np.zeros(len(label_space), dtype=np.float32)
        self.history: list[float] = []
        self.history_lr: list[float] = []

    @staticmethod
    def dataset_fingerprint(pairs: list[dict]) -> str:
        """Hash of the canonical dataset JSON — the same data always yields
        the same fingerprint, whatever the file path."""
        canonical = json.dumps(
            sorted((p["text"], tuple(sorted(p["labels"]))) for p in pairs), sort_keys=True)
        import hashlib
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def fit(self, pairs: list[dict], val_pairs: list[dict] | None = None,
            patience: int | None = None) -> dict:
        X = np.stack([_hash_vec(_tokens(p["text"]), self.dim) for p in pairs])
        Y = np.zeros((len(pairs), len(self.label_space)), dtype=np.float32)
        for i, p in enumerate(pairs):
            for lab in p["labels"]:
                if lab in self.label_space:
                    Y[i, self.label_space.index(lab)] = 1.0
        n = len(pairs)
        X_val = None
        if val_pairs:
            X_val = np.stack([_hash_vec(_tokens(p["text"]), self.dim) for p in val_pairs])
            Y_val = np.zeros((len(val_pairs), len(self.label_space)), dtype=np.float32)
            for i, p in enumerate(val_pairs):
                for lab in p["labels"]:
                    if lab in self.label_space:
                        Y_val[i, self.label_space.index(lab)] = 1.0
        lr_initial = self.lr
        best_val, stale = float("inf"), 0
        stopped_early = False
        epochs_run = 0
        for epoch in range(self.epochs):
            epochs_run = epoch + 1
            # cosine decay to ~0 over the full budget (mirrors lr_scheduler_type="cosine")
            progress = epoch / max(self.epochs - 1, 1)
            lr_t = lr_initial * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
            self.history_lr.append(round(float(lr_t), 6))
            Z = X @ self.W.T + self.b
            P = 1.0 / (1.0 + np.exp(-np.clip(Z, -30, 30)))
            loss = -np.mean(Y * np.log(P + 1e-9) + (1 - Y) * np.log(1 - P + 1e-9))
            grad_w = ((P - Y).T @ X) / n
            grad_b = (P - Y).mean(axis=0)
            self.W -= lr_t * grad_w
            self.b -= lr_t * grad_b
            self.history.append(round(float(loss), 5))
            if X_val is not None and patience is not None:
                Z_val = X_val @ self.W.T + self.b
                P_val = 1.0 / (1.0 + np.exp(-np.clip(Z_val, -30, 30)))
                val_loss = float(np.mean(-Y_val * np.log(P_val + 1e-9)
                                         - (1 - Y_val) * np.log(1 - P_val + 1e-9)))
                if val_loss < best_val - 1e-4:
                    best_val, stale = val_loss, 0
                else:
                    stale += 1
                    if stale >= patience:
                        stopped_early = True
                        break
        stats = {
            "final_loss": self.history[-1],
            "initial_loss": self.history[0],
            "lr_initial": lr_initial,
            "lr_final": self.history_lr[-1],
            "stopped_early": stopped_early,
            "epochs_run": epochs_run,
        }
        if val_pairs:
            preds = [set(self.predict(p["text"])) for p in val_pairs]
            truth = [set(p["labels"]) for p in val_pairs]
            stats["macro_f1"] = per_class_report(truth, preds, labels=self.label_space)["macro_f1"]
        return stats

    def predict(self, text: str, threshold: float = 0.30) -> list[str]:
        vec = _hash_vec(_tokens(text), self.dim)
        z = self.W @ vec + self.b
        probs = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        picks = [self.label_space[i] for i, pr in enumerate(probs) if pr >= threshold]
        return sorted(picks)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label_space": self.label_space,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "history": self.history,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "MultiLabelTrainer":
        payload = json.loads(path.read_text())
        trainer = cls(label_space=payload["label_space"])
        trainer.W = np.asarray(payload["W"], dtype=np.float32)
        trainer.b = np.asarray(payload["b"], dtype=np.float32)
        trainer.history = payload["history"]
        return trainer


def micro_f1(y_true_sets: list[set[str]], y_pred_sets: list[set[str]]) -> dict:
    tp = sum(len(t & p) for t, p in zip(y_true_sets, y_pred_sets))
    fp = sum(len(p - t) for t, p in zip(y_true_sets, y_pred_sets))
    fn = sum(len(t - p) for t, p in zip(y_true_sets, y_pred_sets))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}



def per_class_report(y_true_sets: list[set[str]], y_pred_sets: list[set[str]],
                     labels: list[str]) -> dict:
    """Precision/recall/F1/support per label plus macro-F1 — the per-class
    report every finetune eval ships; a bare micro-F1 hides dead labels."""
    report: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if label in t and label in p)
        fp = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if label not in t and label in p)
        fn = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if label in t and label not in p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                         "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    report["macro_f1"] = round(sum(f1s) / len(f1s), 4) if f1s else 0.0
    return report


class ExperimentLog:
    """Append-only JSONL run log (W&B-lite, honest): config, dataset
    fingerprint (hash of the canonical data, never the file path), and the
    metrics each fit produced. `best()` returns the highest macro-F1 row."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, trainer: MultiLabelTrainer, config: dict,
               dataset: list[dict], stats: dict) -> dict:
        row = {
            "config": dict(config),
            "dataset_fingerprint": MultiLabelTrainer.dataset_fingerprint(dataset),
            "metrics": {
                "initial_loss": stats["initial_loss"],
                "final_loss": stats["final_loss"],
                "macro_f1": stats.get("macro_f1", micro_f1([], [])["f1"]),
            },
            "epochs_run": stats.get("epochs_run"),
            "stopped_early": stats.get("stopped_early", False),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def best(self) -> dict:
        rows = self.rows()
        if not rows:
            raise ValueError("no runs logged yet")
        return max(rows, key=lambda row: row["metrics"]["macro_f1"])
