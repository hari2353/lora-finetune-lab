# LoRA Fine-tune Lab

[![ci](https://github.com/hari2353/lora-finetune-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hari2353/lora-finetune-lab/actions/workflows/ci.yml)

A fine-tuning lab with an honest split: the **real LoRA/PEFT path** (documented, Colab-T4-ready) and a
**deterministic offline twin** that exercises the identical lifecycle — dataset → train → save/load →
eval — on CPU so CI proves the plumbing without a GPU.

## The full-circle story

This lab closes a loop across the stack:

1. [skills-engine](https://github.com/hari2353/skills-engine) defines the skill taxonomy
2. `finetune_lab.data` generates **synthetic instruction pairs** from that taxonomy
3. `train_lora.py` produces the exact TRL/PEFT script for a real run (`Qwen2.5-0.5B-Instruct`, r=16, α=32, q/k/v/o targets)
4. `mock_trainer.py` trains per-skill linear heads offline through the same fit→save→load→predict lifecycle
5. `micro_f1` gates: the trained artifact must beat a random-label baseline in CI

## Offline pipeline (runs anywhere)

```bash
pip install -r requirements.txt
pytest -q    # data determinism, valid SFT records, loss decreases, F1 > random baseline, save/load round-trip
```

## Hygiene layer (market-steal, 2026-11-18)

The offline twin now matches what serious finetune labs treat as baseline:

- **`stratified_split`** — deterministic train/val/test stratified by primary
  skill; splits assign positions, so duplicate rows still land in exactly one
  part; every skill appears in every part.
- **`per_class_report`** — precision/recall/F1/support per label + macro-F1;
  a bare micro-F1 hides dead labels.
- **`ExperimentLog`** — append-only JSONL per run: config, dataset
  fingerprint (hash of the canonical data), macro-F1; `best()` picks the
  winner. Same seed ⇒ same fingerprint and same metrics (tested).
- **Early stop + cosine schedule** — `fit(val_pairs=..., patience=N)` stops
  on val-loss plateau (only after patience is exhausted), decays the LR on a
  cosine over the full budget — mirroring the real path's
  `EarlyStoppingCallback` and `lr_scheduler_type="cosine"` args.

## Training footprint (Harvard Edge cross-pollination)

`finetune_lab.footprint` reports the operational cost of a run beside its F1,
following Harvard Edge's TinyML-Footprint boundary:

```python
training_carbon(gpu_hours=2.0, gpu_power_w=400, pue=1.15, carbon_intensity_g_per_kwh=400)
# {'kwh': 0.92, 'kgco2e': 0.368, ...}

lora_trainable_params(rank=16, modules=[(896, 896)] * 4)   # 114,688 trainable params
lora_share(114_688, 494_032_768)                            # 0.000232 of the base model
```

GPU-hours is the measured input; power, PUE and grid carbon intensity are
explicit assumptions. Embodied hardware is out of scope, matching
TinyML-Footprint's operational boundary.

## Real GPU path (Colab T4, free tier)

See `notebooks/colab_t4.md` for copy-paste cells:

```
pip install torch transformers peft trl datasets
python -m finetune_lab.train_lora   # emits ready-to-run script from DEFAULT_LORA_CONFIG
```

Training args live in one dict (`TRAINING_ARGS`) so offline docs and real runs never drift.

MIT
