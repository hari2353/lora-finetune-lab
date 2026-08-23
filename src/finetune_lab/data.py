from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SKILL_PHRASES = {
    "esco__004": ["kubernetes", "k8s cluster", "pod scheduling", "helm charts"],
    "esco__007": ["python scripting", "pandas pipeline", "async python"],
    "esco__016": ["aws lambda", "s3 buckets", "cloudformation stack"],
    "esco__001": ["machine learning model", "gradient boosting", "feature engineering"],
    "esco__011": ["retrieval augmented generation", "vector search grounding"],
    "esco__005": ["spark dataframe job", "emr cluster tuning"],
}

TEMPLATES = [
    "We need someone with {a} and {b} experience.",
    "Role focuses on {a}, plus familiarity with {b}.",
    "Daily work involves {a} and occasional {b} tasks.",
    "Strong {a} background required; {b} is a plus.",
]


def generate_pairs(n_per_skill: int = 25, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    skill_ids = list(SKILL_PHRASES)
    for sid in skill_ids:
        phrases = SKILL_PHRASES[sid]
        for i in range(n_per_skill):
            a = rng.choice(phrases)
            other_sid = rng.choice([s for s in skill_ids if s != sid])
            b = rng.choice(SKILL_PHRASES[other_sid])
            template = rng.choice(TEMPLATES)
            text = template.format(a=a, b=b)
            expected = [sid]
            if rng.random() < 0.35:
                expected.append(other_sid)
                if rng.random() < 0.5:
                    text = text.replace(b, f"{b} {rng.choice(SKILL_PHRASES[other_sid])}")
            pairs.append({"text": text, "labels": sorted(expected)})
    rng.shuffle(pairs)
    return pairs


def save_csv(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "labels"])
        writer.writeheader()
        for row in pairs:
            writer.writerow({"text": row["text"], "labels": "|".join(row["labels"])})


def load_csv(path: Path) -> list[dict]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append({"text": row["text"], "labels": [x for x in row["labels"].split("|") if x]})
    return out


def to_instruction_format(pairs: list[dict], id_to_name: dict[str, str]) -> list[dict]:
    """OpenAI-style chat records ready for LoRA SFT on a small instruct model."""
    records = []
    for pair in pairs:
        names = sorted(id_to_name.get(sid, sid) for sid in pair["labels"])
        records.append(
            {
                "messages": [
                    {"role": "system", "content": "Extract skills from the text. Reply JSON: {\"skills\": [...] }"},
                    {"role": "user", "content": pair["text"]},
                    {"role": "assistant", "content": '{"skills": ' + json.dumps(names) + "}"}
                ]
            }
        )
    return records


