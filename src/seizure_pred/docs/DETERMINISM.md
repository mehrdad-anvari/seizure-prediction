# Reproducibility & determinism

Use `core/seed.py` to seed python, numpy, and torch.

Training CLI should call `seed_everything(...)` once at startup.
Enabling determinism can reduce performance; disable if you prefer speed.
