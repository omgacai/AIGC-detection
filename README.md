# Robust AIGC Detection — Phase 0

Phase 0 provides a reproducible local-to-cluster foundation for the hackathon task: distinguish authentic images (`0`) from AI-generated images (`1`). It deliberately does **not** implement a detector, DINO training, robust augmentations, or any Phase 1 method.

## Guardrails

- Datasets, checkpoints, outputs, logs, Hugging Face caches, and secrets are ignored by Git.
- `organizer_demo` records are rejected by `AIGCImageDataset(..., for_training=True)`. The organiser COCO/DALL·E Advanced demonstration data must never be used for training or model selection.
- Dataset manifests have `path,label,source_dataset,generator,split` fields. Split isolation is verified before use.
- Storage is resolved in this order: `AIGC_*` environment variable, `configs/paths.yaml`, then repo-local defaults.

## Local setup

```bash
git clone https://github.com/omgacai/AIGC-detection.git
cd AIGC-detection
bash scripts/setup_env.sh
cp configs/paths.example.yaml configs/paths.yaml
python scripts/check_gpu.py
pytest -q
```

Python 3.10–3.12 is supported. On a Mac, CUDA being unavailable is expected. It is a diagnostic, not a failure of the code.

## Cluster setup

Use Git for source code, but put all large artefacts on cluster-accessible storage:

```bash
export AIGC_DATA_ROOT=/path/to/cluster/storage/aigc_data
export AIGC_CACHE_ROOT=/path/to/cluster/storage/aigc_cache
export AIGC_VENV_DIR="$AIGC_CACHE_ROOT/venvs/robust-aigc"
export AIGC_CHECKPOINT_ROOT=/path/to/cluster/storage/aigc_checkpoints
export AIGC_OUTPUT_ROOT=/path/to/cluster/storage/aigc_outputs
export HF_HOME="$AIGC_CACHE_ROOT/huggingface"
export TORCH_HOME="$AIGC_CACHE_ROOT/torch"
git pull
VENV_DIR="$AIGC_VENV_DIR" bash scripts/setup_env.sh
```

`setup_env.sh` preserves an already available PyTorch installation rather than reinstalling it. If no PyTorch is available, it installs the declared dependencies and prints a CUDA diagnostic. On a cluster, confirm the resulting build works inside a GPU allocation:

```bash
python scripts/check_gpu.py
```

## First dataset: SID_Set

Run this **on the cluster**, after checking available storage. It uses `huggingface_hub` and honours `HF_HOME`; it does not use a Mac dataset directory or `~/.cache` by default.

```bash
df -h "$AIGC_DATA_ROOT"
python scripts/download_datasets.py --dataset sid --output-dir "$AIGC_DATA_ROOT"
export AIGC_MANIFEST="$AIGC_DATA_ROOT/manifests/sid_all.csv"
python scripts/verify_dataset.py --manifest "$AIGC_MANIFEST"
python scripts/inspect_dataset.py --dataset sid --manifest "$AIGC_MANIFEST"
python scripts/dataloader_smoke_test.py --manifest "$AIGC_MANIFEST"
```

The scanner accepts common folder names such as `real`, `authentic`, `fake`, `ai`, and `generated`. If SID_Set has a different layout, the download completes but manifest generation stops with an actionable layout message instead of guessing labels.

`CIFAKE` remains available as a quick low-resolution pipeline check when Kaggle credentials are configured; it is not the preferred robustness dataset. `WildFake` is deliberately manual until its ModelScope layout and overlap with organiser data are verified.

## DINOv3 availability only

No DINOv3 training occurs in Phase 0. The default is Meta's DINOv3 ViT-L/16 checkpoint, `facebook/dinov3-vitl16-pretrain-lvd1689m` (about 300M parameters): comfortably under the 2B limit and much more practical than the 7B teacher for the observed 12 GB TITAN V nodes.

After the data smoke test, accept the DINOv3 licence on Hugging Face and authenticate with `hf auth login` (never commit the token), then run inside a GPU job:

```bash
python scripts/check_dinov3.py
```

It loads DINOv3 ViT-L/16, verifies the checkpoint has fewer than 2B parameters, runs one mixed-precision forward pass, reports feature shape and GPU memory, releases memory, and suggests DINOv3 ViT-B/16 (about 86M) only on out-of-memory.

## Slurm

See [cluster/README.md](cluster/README.md). The Slurm scripts intentionally use generic placeholder directives. Set the required partition, account, QoS, and exact GPU GRES after inspecting the target cluster; do not assume names across clusters.

```bash
sbatch cluster/slurm_smoke_test.sh
```

## Future training outputs

The training utilities are ready without introducing a model. Future `train.py` should write each epoch's loss/validation metrics through `EpochMetricsWriter`, which creates `metrics.jsonl` and `metrics.csv` under `$AIGC_OUTPUT_ROOT/<run-name>/`. `CheckpointManager` atomically updates `$AIGC_CHECKPOINT_ROOT/<run-name>/last.pt` each epoch and writes `best.pt` only when the configured internal-validation metric improves. This preserves a resumable state with model, optimizer, scheduler, epoch, metrics, and training arguments.

Never put needed checkpoints in `$SLURM_TMPDIR`; it is node-local temporary storage and can disappear after the Slurm job.

## Status boundary

Stop after one real dataset is downloaded, manifest-backed, verified, previewed, and produces a DataLoader batch; then validate DINOv3 on GPU. Robust transforms, paired learning, feature losses, model training, and tuning are intentionally deferred to Phase 1.
