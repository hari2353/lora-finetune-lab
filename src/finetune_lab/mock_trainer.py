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
    trained with SGD. Same lifecycle as the real path (fit -> save -> load -> predict)."""

    def __init__(self, label_space: list[str], dim: int = 512, lr: float = 0.6, epochs: int = 80) -> None:
        self.label_space = label_space
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.W = np.zeros((len(label_space), dim), dtype=np.float32)
        self.b = np.zeros(len(label_space), dtype=np.float32)
        self.history: list[float] = []

    def fit(self, pairs: list[dict]) -> dict:
        X = np.stack([_hash_vec(_tokens(p["text"]), self.dim) for p in pairs])
        Y = np.zeros((len(pairs), len(self.label_space)), dtype=np.float32)
        for i, p in enumerate(pairs):
            for lab in p["labels"]:
                if lab in self.label_space:
                    Y[i, self.label_space.index(lab)] = 1.0
        n = len(pairs)
        for _ in range(self.epochs):
            Z = X @ self.W.T + self.b
            P = 1.0 / (1.0 + np.exp(-np.clip(Z, -30, 30)))
            loss = -np.mean(Y * np.log(P + 1e-9) + (1 - Y) * np.log(1 - P + 1e-9))
            grad_w = ((P - Y).T @ X) / n
            grad_b = (P - Y).mean(axis=0)
            self.W -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.history.append(round(float(loss), 5))
        return {"final_loss": self.history[-1], "initial_loss": self.history[0]}

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

