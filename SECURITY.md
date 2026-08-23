# Security Notes

## Controls
- Synthetic data only: no real PII exists anywhere in datasets or tests
- Training artifacts are JSON weights — no pickle/code-execution load path
- Real-GPU script is generated code run by the operator, never auto-executed on import

## Known gaps
- `train_lora.py` output assumes a trusted notebook environment; do not paste untrusted configs
