# Cluster workflow

Keep source code in Git and datasets, caches, checkpoints, outputs, and logs on cluster-accessible storage. Do not commit any of those artefacts.

```bash
# Local Mac
git add . && git commit -m "Phase 0 infrastructure" && git push

# Cluster login node
git clone <REPOSITORY_URL> robust-aigc
cd robust-aigc
cp configs/paths.example.yaml configs/paths.yaml
export AIGC_DATA_ROOT=/path/to/large/cluster/storage/aigc_data
export AIGC_CACHE_ROOT=/path/to/large/cluster/storage/aigc_cache
export HF_HOME="$AIGC_CACHE_ROOT/huggingface"
export TORCH_HOME="$AIGC_CACHE_ROOT/torch"
bash scripts/setup_env.sh
```

Before downloading, check capacity with `df -h "$AIGC_DATA_ROOT"`. Download the first dataset only on cluster storage:

```bash
python scripts/download_datasets.py --dataset sid --output-dir "$AIGC_DATA_ROOT"
export AIGC_MANIFEST="$AIGC_DATA_ROOT/manifests/sid_all.csv"
python scripts/verify_dataset.py --manifest "$AIGC_MANIFEST"
python scripts/inspect_dataset.py --dataset sid --manifest "$AIGC_MANIFEST"
```

For Slurm, inspect `sinfo` and `scontrol show partition`, adjust the placeholder directives in `slurm_smoke_test.sh`, then submit with `sbatch cluster/slurm_smoke_test.sh`. The SoC cluster observed earlier needs a specific GRES name such as `gpu:nv:1`; do not copy that blindly to another cluster. Login nodes normally have no GPU, so run GPU checks only in the job.

Never place credentials, tokens, or private keys in this repository. SSH jump hosts belong in your personal `~/.ssh/config`, for example:

```sshconfig
Host gpu-cluster
    HostName <GPU_HOST>
    User <USERNAME>
    ProxyJump <JUMP_HOST>
    IdentityFile ~/.ssh/id_ed25519
```
