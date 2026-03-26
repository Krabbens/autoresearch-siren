# autoresearch-siren — Project Context

## Project Overview

**autoresearch-siren** is an autonomous experimentation harness for training **SIREN V8 VQ-VAE Phase 1** — a neural audio codec with residual FSQ bottlenecks for semantic, prosody, and speaker branches. It is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted to work with the sibling `../SIREN` repository.

**Purpose:** Enable rapid, time-budgeted training experiments where an AI agent can freely modify the architecture and training pipeline in `train.py` to discover better designs for reconstructing frozen BitHuBERT features through a factorizer → Residual FSQ bottlenecks → feature reconstructor.

**Key Philosophy:** The SIREN codebase is **not sacred** — agents are explicitly encouraged to edit, replace, or discard any part of the architecture in `train.py` to explore better designs.

## Repository Structure

```
autoresearch-siren/
├── prepare.py          # Fixed harness: constants, asset verification, evaluation helpers (DO NOT EDIT)
├── train.py            # Agent-editable: entire model + training pipeline lives here
├── program.md          # Detailed experimentation protocol and agent instructions
├── pyproject.toml      # Project dependencies (editable siren-codec from ../SIREN)
├── results.tsv         # Experiment log (tab-separated, untracked)
├── run_*.log           # Training logs from experiments
├── analysis.ipynb      # Upstream artifact (optional)
├── progress.png        # Upstream artifact (optional)
└── checkpoints/
    └── vqvae_autoresearch/
        └── vqvae_autoresearch_last.pt  # Latest weights
```

## Dependencies & Requirements

- **Python ≥ 3.11** (required for `onnxruntime` wheels)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **CUDA GPU** recommended (same setup as training SIREN)
- **Sibling checkout:** `../SIREN` (editable dependency `siren-codec`)

## Building and Running

### Setup

```bash
cd autoresearch-siren
uv sync
uv run prepare.py    # Verify wavs, config, checkpoint
```

### Training

```bash
uv run train.py      # Trains until prepare.TIME_BUDGET (default 300s wall after warmup)
```

### Smoke Test

```bash
SIREN_TIME_BUDGET=30 uv run train.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIREN_ROOT` | `../SIREN` | Path to sibling SIREN checkout |
| `SIREN_DATA_DIR` | `../SIREN/data/waves_recovered_16k` | Training wav directory |
| `SIREN_CONFIG` | `.../ultra58bps_16k.yaml` | VQ-VAE config path |
| `SIREN_HUBERT_CKPT` | `.../bithubert_best.pt` | BitHuBERT checkpoint |
| `SIREN_TIME_BUDGET` | `300` | Wall-clock training budget (seconds) |
| `SIREN_SKIP_SPEECH_METRICS` | `0` | Skip slow Griffin-Lim eval |
| `SIREN_SPEECH_METRICS_CLIPS` | `12` | Number of clips for speech metrics |
| `SIREN_GL_ITER` | `24` | Griffin-Lim iterations |

## Key Metrics

Training outputs a summary block after `---`:

| Metric | Direction | Description |
|--------|-----------|-------------|
| `val_score` | **Lower** | Primary metric: recon MSE + FSQ collapse penalties |
| `val_recon_mse` | **Lower** | Mean reconstruction MSE on validation |
| `val_pesq_wb` | **Higher** | Wideband PESQ (ITU-T P.862.2) |
| `val_stoi` | **Higher** | STOI (Taal et al.) |
| `sem_h_bits` | — | Semantic branch entropy (bits/tok) |
| `pro_h_bits` | — | Prosody branch entropy |
| `spk_h_bits` | — | Speaker branch entropy |
| `peak_vram_mb` | — | Peak GPU memory |

**FSQ Collapse Detection:** Large penalties added to `val_score` if any branch shows entropy < threshold with low codebook usage.

## Development Conventions

### File Boundaries

- **`prepare.py`** — **DO NOT EDIT**: Contains fixed constants (`TIME_BUDGET`, paths), asset verification, and evaluation helpers (`evaluate_vqvae_val`, `measure_speech_pesq_stoi`, `compute_val_score`).
- **`train.py`** — **EDIT FREELY**: The only agent-editable file. Contains the entire model architecture, training loop, losses, and logging.

### Experimentation Workflow

1. Work on branch `autoresearch/<tag>` (e.g., `autoresearch/mar26`)
2. Edit `train.py` with one hypothesis
3. Commit: `git commit -am "description"`
4. Run: `uv run train.py > run.log 2>&1`
5. Extract metrics: `grep "^val_score:" run.log`
6. Log to `results.tsv` (tab-separated):
   ```
   commit	val_score	memory_gb	status	description
   abc1234	0.452100	11.0	keep	baseline
   ```
7. If improved → keep commit; if worse → `git reset`

### results.tsv Format

```tsv
commit	val_score	memory_gb	status	description
a1b2c3d	0.452100	11.0	keep	baseline
b2c3d4e	0.441200	11.2	keep	higher quant_loss_weight in train.py
c3d4e5f	15.200000	11.0	discard	pro FSQ collapsed; penalty dominated val_score
d4e5f6g	0.000000	0.0	crash	OOM after widening fusion
```

**Status values:** `keep`, `discard`, `crash`

### Coding Style

- **Architecture freedom:** Swap blocks, reimplement layers, change losses, replace FSQ, modify HuBERT feature usage
- **Validation requirement:** Every experiment must run to completion and log results
- **Metric consistency:** Keep printing `val_score:`, `val_recon_mse:`, etc. prefixes for grep compatibility
- **Simplicity preference:** Small changes when sufficient; large rewrites allowed when SIREN structure is the problem

## Architecture Components (Default)

The default `train.py` implements:

1. **BitHuBERT** — Frozen feature extractor (loaded from checkpoint)
2. **InformationFactorizerV2** — Splits features into semantic/prosody/speaker branches
3. **Residual FSQ** — Three separate quantizers (semantic, prosody, speaker)
4. **FeatureReconstructorV2** — Reconstructs HuBERT features from quantized branches

**Loss components:**
- Reconstruction MSE (primary)
- Quantization loss (weighted per branch)
- Spread penalty (encourages codebook usage)
- Latent diversity term (variance regularization)
- Batch Gram logdet (optional, for diversity)

## Training Loop Details

- **Warmup:** First `WARMUP_TRAINING_STEPS=10` steps don't count toward time budget
- **LR schedule:** Linear warmup (5%), constant, then cosine decay (10%)
- **AMP:** Mixed precision enabled by default on CUDA
- **Gradient accumulation:** Configurable via `grad_accum_steps`
- **Checkpoint:** Saved to `checkpoints/vqvae_autoresearch/vqvae_autoresearch_last.pt`

## Speech Quality Metrics (Proxy)

`val_pesq_wb` and `val_stoi` are computed via:
1. Fixed linear HuBERT→mel projection (frozen, seed=42)
2. InverseMelScale + Griffin-Lim (torchaudio)
3. Compare reconstructed waveform to original

**Important:** These are **proxy metrics** for tracking training, not full neural codec E2E evaluation.

## Common Failure Modes

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| `val_score` >> 10 | FSQ collapse (entropy ~0) | Increase diversity/spread weights |
| OOM | Batch too large | Reduce batch, increase grad_accum |
| `nan` loss | Instability | Reduce LR, check gradient clipping |
| Slow iteration | Speech metrics | Set `SIREN_SKIP_SPEECH_METRICS=1` |
