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

## Real GPU path (Colab T4, free tier)

See `notebooks/colab_t4.md` for copy-paste cells:

```
pip install torch transformers peft trl datasets
python -m finetune_lab.train_lora   # emits ready-to-run script from DEFAULT_LORA_CONFIG
```

Training args live in one dict (`TRAINING_ARGS`) so offline docs and real runs never drift.

MIT
