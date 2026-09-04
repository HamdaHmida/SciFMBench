# SciFMBench

A unified benchmarking framework for **fluid-dynamics Scientific Foundation Models (SciFMs)**.

Open-source SciFMs ship with weights and code, but each has its own data format,
input modality, and pre/post-processing pipeline. SciFMBench wraps every model
behind a single interface so they can be trained, finetuned, and evaluated on
equal footing — apples-to-apples.

## Design at a glance

```
┌─────────────────────────────────────────────────────────┐
│  Engine / CLI      train | finetune | test              │  ← you run this
├─────────────────────────────────────────────────────────┤
│  Benchmark layer   shared splits, normalization,        │  ← fair comparison
│                   canonical metrics                     │
├─────────────────────────────────────────────────────────┤
│  Adapter (per model)  thin wrapper, no upstream edits   │  ← the protocol
├─────────────────────────────────────────────────────────┤
│  Vendored model    MPP, FNO, DeepONet, ... (untouched)  │
└─────────────────────────────────────────────────────────┘
```

**Core idea — the adapter pattern.**

Each model ships in `models/<name>/` with two files:

- `model.py` — the vendored upstream code, **unmodified**.
- `adapter.py` — a thin wrapper implementing the `BaseModel` ABC.

The framework never touches upstream code. When the original repo updates, you
bump a pin and the adapter either still works or needs a small update. No forks
to maintain.

**Native training, canonical evaluation (Option C).**

- **Training** is delegated to each model's native loop — each model trains the
  way its authors designed. The framework doesn't impose optimizer, scheduler,
  or loss.
- **Evaluation** is canonical — shared splits, shared normalization, shared
  metrics — so differences in results reflect the model, not the wrapper.

---

## Install

```bash
# 1. Clone
git clone <this-repo> && cd SciFMBench

# 2. (Recommended) create a fresh environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The framework imports are lazy for `torch` and `numpy` in utility modules, so
> `python -m engine.run --list-models` works even before installing deps.
> Anything that actually builds or runs a model requires `torch`.

---

## Run

### List registered models

```bash
python -m engine.run --list-models
```

### Train from scratch (uses `basic_config` section by default)

```bash
python -m engine.run \
    --model MPP \
    --mode train \
    --config configs/models/mpp_avit_s_config.yaml
```

### Finetune (uses `finetune` section; loads `pretrained_ckpt_path`)

```bash
python -m engine.run \
    --model MPP \
    --mode finetune \
    --config configs/models/mpp_avit_s_config.yaml \
    --weights /path/to/ckpt.tar
```

### Zero-shot eval

```bash
python -m engine.run \
    --model MPP \
    --mode test \
    --config configs/models/mpp_avit_s_config.yaml \
    --weights /path/to/ckpt.tar
```

### Pick a config section explicitly

```bash
python -m engine.run --model MPP --mode train \
    --config configs/models/mpp_avit_s_config.yaml \
    --section frozen          # or basic_config | finetune | less_frozen
```

Or let the CLI pick by inspecting the section's `pretrained` flag:

```bash
python -m engine.run --model MPP --mode finetune \
    --config configs/models/mpp_avit_s_config.yaml \
    --auto-section
```

The script also accepts JSON configs (the loader falls back when PyYAML is
unavailable).

---

## Project layout

```
SciFMBench/
├── configs/                # YAML / JSON experiment configs
│   ├── models/             # per-model hyperparameters
│   └── datasets/           # dataset paths and splits (placeholder)
│
├── core/                   # Abstract Base Classes — the framework's contracts
│   ├── base_model.py       #   BaseModel: build / forward / to_canonical / train / ...
│   ├── base_processor.py   #   BaseProcessor: pre/post-processing
│   └── base_metric.py      #   BaseMetric: shared evaluation metrics
│
├── data/                   # Data loading and shared transforms
│   └── synthetic.py        #   synthetic (T,B,C,H,W) batches — smoke-test loader
│
├── models/                 # One folder per model
│   ├── my_model/           #   adapter pattern scaffold (placeholder)
│   └── MPP/                #   vendored Axial-ViT-for-PDE + thin adapter
│       ├── avit.py         #     ← vendored, untouched
│       ├── spatial_modules.py
│       ├── time_modules.py
│       ├── mixed_modules.py
│       ├── shared_modules.py
│       └── adapter.py      #   ← the only file we author
│
├── processing/             # Model-specific pre/post processors
│
├── engine/                 # Training and evaluation loops
│   └── run.py              #   the CLI entry point
│
├── utils/                  # Cross-cutting utilities
│   ├── config.py           #   YAML / JSON config loader
│   └── seed.py             #   global seeding
│
├── run.py                  # top-level launcher (equivalent to engine.run)
├── requirements.txt
└── README.md
```

---

## How a new model gets added

1. Drop the upstream repo into `models/<name>/` as `model.py` (or a folder).
2. Write `models/<name>/adapter.py` implementing `BaseModel`:
   - `build(cfg)` — instantiate the vendored model.
   - `forward(x)` — single-arg wrapper around the upstream forward.
   - `to_canonical` / `from_canonical` — wrap outputs for the evaluator.
   - `load_weights` / `save_weights` — local path; extend for HF Hub / URL.
   - `train(...)` — call the upstream training script.
3. Decorate the adapter with `@register` (already imported by
   `models/<name>/__init__.py`).
4. The model now appears in `python -m engine.run --list-models`.

The framework never edits vendored code. Updates flow through by re-syncing
the upstream folder.

---

## Status

| Task | Status |
|---|---|
| BaseModel ABC + adapter protocol | ✅ done |
| First adapter (MPP / Axial ViT for PDE) | ✅ done |
| CLI training / eval script | ✅ done — supports train / finetune / test, MPP YAML format |
| Canonical Sample schema for fluid dynamics | ⏳ next |
| Real dataset loader (PDEBench paths from the config) | ⏳ pending |
| Evaluation / benchmark layer (canonical eval) | ⏳ pending |
| Full training runner (logging, checkpoints, AMP) | ⏳ pending |

Currently the framework ships with **synthetic data** so the wiring path can
be smoke-tested without a real PDEBench install. To exercise the upstream
data paths in `configs/models/mpp_avit_s_config.yaml` (PDEBench 2D shallow-water,
incompressible NS, compressible NS, diffusion-reaction), the dataset loader
(task #5) needs to land first.

---

## License

TBD.
