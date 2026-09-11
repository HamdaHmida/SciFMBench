# SciFMBench

A unified benchmarking framework for **fluid-dynamics Scientific Foundation Models (SciFMs)**.

Open-source SciFMs ship with weights and code, but each has its own data format,
input modality, and pre/post-processing pipeline. SciFMBench wraps every model
behind a single interface so they can be trained, finetuned, and evaluated on
equal footing — apples-to-apples.

## Design at a glance
**Core idea — the adapter pattern.**

Each model ships in `models/<name>/` with two files:

- `model.py` — the vendored upstream code, **unmodified**.
- `adapter.py` — a thin wrapper implementing the `BaseModel` ABC.

The framework never touches upstream code. When the original repo updates, you
bump a pin and the adapter either still works or needs a small update. No forks
to maintain.

**Native training, canonical evaluation.**

- **Training** is delegated to each model's native loop — each model trains the
  way its authors designed. The framework doesn't impose optimizer, scheduler,
  or loss.
- **Evaluation** is canonical — shared splits, shared normalization, shared
  metrics — so differences in results reflect the model, not the wrapper.

**Canonical data format (the framework's contract for fluid dynamics):**

```
canonical = (B, Nx, Ny, T, C) float32
```

| Axis | Meaning |
|---|---|
| `B`  | batch size |
| `Nx` | grid points along x |
| `Ny` | grid points along y |
| `T`  | time-window length |
| `C`  | physical state channels (e.g. vx, vy, pressure, density) |

Each adapter implements `to_canonical` / `from_canonical` to translate
between its model's native layout and this canonical form. The canonical
form is a pure permutation of typical upstream formats: e.g. MPP's
native `(T, B, C, Nx, Ny)` becomes `(B, Nx, Ny, T, C)` via
`permute(1, 3, 4, 0, 2)`.

---

## Install

```bash
# 1. Clone
git clone https://github.com/HamdaHmida/SciFMBench.git && cd SciFMBench

# 2. (Recommended) create a fresh environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The framework imports are lazy for `torch` and `numpy` in utility modules, so
> `python3 -m engine.run --list-models` works even before installing deps.
> Anything that actually builds or runs a model requires `torch`.

---

## Run

### List registered models

```bash
python3 -m engine.run --list-models
```

### Train from scratch (uses `basic_config` section by default)

```bash
python3 -m engine.run \
    --model <modelname> \
    --mode train \
    --config configs/models/model_config.yaml
```

### Finetune (uses `finetune` section; loads `pretrained_ckpt_path`)

```bash
python3 -m engine.run \
    --model <modelname> \
    --mode finetune \
    --config configs/models/model_config.yaml \
    --weights ../weights/model_weights.pt
```

### Zero-shot eval

```bash
python3 -m engine.run \
    --model <modelname> \
    --mode test \
    --config configs/models/model_config.yaml \
    --weights ../weights/model_weights.pt
```

### Pick a config section explicitly

```bash
python3 -m engine.run --model MPP --mode train \
    --config configs/models/mpp_avit_s_config.yaml \
    --section frozen          # or basic_config | finetune | less_frozen
```

Or let the CLI pick by inspecting the section's `pretrained` flag:

```bash
python3 -m engine.run --model MPP --mode finetune \
    --config configs/models/mpp_avit_s_config.yaml \
    --auto-section
```

The script also accepts JSON configs (the loader falls back when PyYAML is
unavailable).

### Smoke-test a model before a real run

Before launching a full training run, you can quickly verify that a model is
wired up correctly. `engine/test_models.py` generates random canonical data
of shape `(16, 128, 128, 20, 2)`, runs 1 training epoch over a `DataLoader`
with `batch_size=4`, and reports PASS / FAIL on:

- model builds without error
- forward output shape matches the expected native output
- `to_canonical` / `from_canonical` round-trip cleanly
- training loop completes
- losses are finite and in a sane range
- gradients are flowing (loss isn't collapsed or exploded)

```bash
python3 -m engine.test_models --model MPP
python3 -m engine.test_models --list-models
```

Exit code is 0 on PASS, 1 on FAIL — useful for CI / pre-flight checks.

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
│   └── <modelname>/        #   vendored Axial-ViT-for-PDE + thin adapter
│       ├── model.py         #   ← vendored, untouched
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
4. The model now appears in `python3 -m engine.run --list-models`.

The framework never edits vendored code. Updates flow through by re-syncing
the upstream folder.

---

## Status

| Task | Status |
|---|---|
| BaseModel ABC + adapter protocol | ✅ done |
| First adapter (MPP / Axial ViT for PDE) | ✅ done |
| CLI training / eval script | ✅ done — supports train / finetune / test, MPP YAML format |
| Smoke-test script (`engine/test_models.py`) | ✅ done |
| Canonical Sample schema for fluid dynamics | ✅ done |
| Real dataset loader (PDEBench paths from the config) | ⏳ pending |
| Evaluation / benchmark layer (canonical eval) | ✅ done |
| Add Different models with their adapters | ⏳ pending |

Currently the framework ships with **synthetic data** so the wiring path can
be smoke-tested without a real PDEBench install. To exercise the upstream
data paths in `configs/models/mpp_avit_s_config.yaml` (PDEBench 2D shallow-water,
incompressible NS, compressible NS, diffusion-reaction), the dataset loader
(task #5) needs to land first.

---

## Weights

Model checkpoints are **not** stored in the repo — GitHub rejects files over
100 MB, and SciFM weights routinely exceed that. The convention is:

```
<parent>/                # e.g. /home/hamda/PhD_Thesis/
├── SciFMBench/          # the repo
└── weights/             # checkpoints live here, sibling of the repo
    ├── mpp_finetune.pt
    └── mpp_pretrained.tar
```

From inside the repo, point `--weights` at the sibling folder:

```bash
python3 -m engine.run --model MPP --mode test \
    --config configs/models/mpp_avit_s_config.yaml \
    --weights ../weights/mpp_pretrained.tar
```

The repo's `.gitignore` excludes any in-repo `weights/` folder so a stray
local copy doesn't sneak into a commit. The `weights/` directory itself ships
with a `README.md` describing the convention.

---

## Datasets

Raw dataset files follow the same rule as weights — they live in a sibling
folder, not in the repo. GitHub rejects large files, and PDEBench HDF5
files routinely exceed that.

```
<parent>/                # e.g. /home/hamda/PhD_Thesis/
├── SciFMBench/          # the repo
├── weights/             # checkpoints
└── datasets/            # raw dataset files live here
    └── PDEBench/
        └── 2D/
            ├── shallow-water/
            ├── NS_incom/
            ├── CFD/
            └── diffusion-reaction/
```

The MPP config (`configs/models/mpp_avit_s_config.yaml`) already points
`train_data_paths` and `valid_data_paths` at `../datasets/PDEBench/...`.
You just need to drop the actual HDF5 files into the matching subfolders.

> **The `data/` Python package inside the repo is unrelated.** That's
> framework code (data loaders, transforms, the synthetic smoke-test
> generator). It stays in the repo. The external `datasets/` folder is
> only for raw data files.

---

## License

TBD.