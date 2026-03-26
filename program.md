# autoresearch-siren

Autonomous experimentation on **SIREN V8 VQ-VAE Phase 1**: reconstruct frozen BitHuBERT features through a factorizer, **Residual FSQ** bottlenecks (semantic / prosody / speaker), and a feature reconstructor. This repo is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) wired to the **SIREN** codec codebase via an editable install of the sibling `../SIREN` repository.

## Important: SIREN code is not sacred

Treat the **SIREN** package (`ultra_low_bitrate_codec`, configs, checkpoints) as **convenience defaults only**, not as a reference implementation you must preserve. The upstream SIREN codebase is **not assumed to be correct or working well**; it may be buggy, unstable, or poorly matched to this harness.

**You are explicitly encouraged** — inside **`train.py` only** — to **edit, replace, or throw away** any part of the architecture and training pipeline that came from SIREN: swap blocks, reimplement layers inline, change losses, replace FSQ with something else, change how HuBERT features are used, collapse or split branches, etc. The starting `train.py` is a **starting point**, not a spec. If you invent a **better** design, **run it** (`uv run train.py`), **read the log**, and **record the outcome in `results.tsv`** (commit, `val_score`, memory, status, description) like any other experiment. If your stack no longer matches `prepare.evaluate_vqvae_val` / `measure_speech_pesq_stoi`, adapt the calls in `train.py` or compute analogous scalars, but **keep printing the same `---` summary line prefixes** (`val_score:`, `val_recon_mse:`, …) whenever they still make sense so `grep` and the TSV workflow stay usable; use `nan` or note divergence in the **description** column if a field no longer applies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `master`.
3. **Read the in-scope files**:
   - `README.md` — paths, `uv`, sibling `SIREN` checkout.
   - `prepare.py` — **fixed** constants (`TIME_BUDGET`, default paths), asset checks, **`evaluate_vqvae_val`** (validation recon MSE + FSQ code stats + `val_score`), and **`measure_speech_pesq_stoi`** — objective **speech** metrics from the literature: **wideband PESQ** (ITU-T P.862 family, via the `pesq` package) and **STOI** (Taal et al., via `pystoi`). A short val waveform is built from `h_recon` with a **frozen** linear HuBERT→mel map + **InverseMelScale + Griffin–Lim** (torchaudio), so scores track training but are a **proxy**, not a full neural codec E2E test. **Do not modify.**
   - `train.py` — **the only file you edit**: the **entire** model + training pipeline may live here (see *SIREN code is not sacred* above).
4. **Verify assets**: Run `uv run prepare.py`. It checks `.wav` data, YAML config, and BitHuBERT checkpoint (defaults point at the sibling `../SIREN` tree; override with env vars in `prepare.py` if needed).
5. **Initialize `results.tsv`**: create it with **only** the header row. The baseline is recorded after the first run.
6. **Confirm and go**: confirm setup, then start the experiment loop.

## Experimentation

Each experiment runs on **one GPU** (CPU works but is slow). Training runs for a **fixed wall-clock budget** after a short step warmup (see `prepare.TIME_BUDGET`, default 300 seconds; overridable with env `SIREN_TIME_BUDGET` for smoke tests). Launch:

```bash
uv run train.py > run.log 2>&1
```

**What you CAN do**

- Modify **`train.py` only** — **full freedom** within that file: architecture, data flow, losses, optimizers, schedules, logging, and how you invoke (or replace) evaluation. You are **not** restricted to “small tweaks inside the V8 stack.”

**What you CANNOT do**

- Modify **`prepare.py`**. It holds the time budget, default paths, and the **fixed** validation harness (`evaluate_vqvae_val`).
- Add new PyPI dependencies beyond what resolves from this repo's `pyproject.toml` / editable `siren-codec` (which already includes `pesq` / `pystoi` as direct deps of this fork).
- Change how `val_score` is computed inside `prepare.py` (that is the ground-truth metric for logging).

**Goal**

- **Lower `val_score`** (printed after `---`), subject to **healthy FSQ usage**.
- `val_score` = `val_recon_mse` plus **large penalties** if semantic/prosody/speaker branches show **collapse** (see `prepare.compute_val_score`). Treat collapse as a failed run even if MSE looks good.
- Also inspect **`sem_h_bits`**, **`pro_h_bits`**, **`spk_h_bits`**, **`val_pesq_wb`**, **`val_stoi`**, and the **train** “Code stats” block. **Higher** is better for PESQ and STOI (unlike `val_score`). Do not conclude stability from a single early snapshot: in full SIREN training, epoch 1 can look misleading; watch for **entropy collapse** over consecutive runs.
- To skip slow Griffin–Lim during iteration: set env **`SIREN_SKIP_SPEECH_METRICS=1`**. Tune clip count with **`SIREN_SPEECH_METRICS_CLIPS`** and GL iterations with **`SIREN_GL_ITER`**.

**VRAM**

- Soft constraint: stay within reasonable GPU memory for the default config; OOM → log `crash`, revert, try smaller batch or more grad accumulation in `train.py`.

**Simplicity vs bold refactors**

- Prefer smaller changes when they suffice. **Large rewrites are allowed** when you believe SIREN’s structure is the problem — but you must **validate** with a real run and **`results.tsv`**. A big simplification that matches or beats metrics is a strong win.

**First run**

- Always establish a **baseline** with unmodified `train.py` (after any human setup commits), then iterate.

## Output format

When the script finishes it prints a summary like:

```
---
val_score:          0.123456
val_recon_mse:      0.123456
val_pesq_wb:        2.345678
val_stoi:           0.789012
speech_metrics_n:   12
sem_h_bits:         2.5000
pro_h_bits:         1.2000
spk_h_bits:         3.0000
training_seconds:   300.2
total_seconds:      330.5
peak_vram_mb:       12000.0
num_steps:          142
checkpoint:         .../vqvae_autoresearch_last.pt
```

Extract the primary metric:

```bash
grep "^val_score:" run.log
grep "^val_pesq_wb:\|^val_stoi:" run.log
```

## Logging results

Append rows to **`results.tsv`** (tab-separated, **not** CSV — commas break descriptions).

Header and **five** columns:

```
commit	val_score	memory_gb	status	description
```

1. Git commit hash (short, 7 chars).
2. **`val_score`** from the run (use `0.000000` for crashes).
3. Peak memory in GB, one decimal (`peak_vram_mb / 1024`); `0.0` on crash.
4. `keep`, `discard`, or `crash`.
5. Short description of the experiment — **especially** if you changed architecture or diverged from the default SIREN stack (say so explicitly).

Example:

```
commit	val_score	memory_gb	status	description
a1b2c3d	0.452100	11.0	keep	baseline
b2c3d4e	0.441200	11.2	keep	higher quant_loss_weight in train.py
c3d4e5f	15.200000	11.0	discard	pro FSQ collapsed; penalty dominated val_score
d4e5f6g	0.000000	0.0	crash	OOM after widening fusion
```

**Do not commit `results.tsv`** (leave untracked).

## Experiment loop

On a dedicated branch (e.g. `autoresearch/mar26`):

**LOOP:**

1. Note current branch/commit.
2. Edit **`train.py`** with one hypothesis.
3. `git commit`.
4. `uv run train.py > run.log 2>&1` (redirect all output; do not flood the terminal).
5. Read results: `grep "^val_score:\|^peak_vram_mb:\|^val_recon_mse:" run.log`. Empty → crash; use `tail -n 80 run.log` for the traceback.
6. Append a row to `results.tsv` — **every** finished run (including radical architecture experiments) gets a row so the trajectory is auditable.
7. If **`val_score` improved (lower)** (or your agreed primary metric improved) and you are satisfied with side metrics / stability, keep the commit (advance).
8. If equal/worse (or collapse), `git reset` to the previous best.

**Timeout**

- Expect ~`TIME_BUDGET` wall time plus model load and final eval. If a run **exceeds ~2× TIME_BUDGET** without finishing, kill it, log `crash`, revert.

**Crashes**

- Trivial fixes (typo, shape) → fix and rerun.
- Bad idea → log `crash`, move on.

**Autonomy**

- After setup, do not ask the human whether to continue the loop. Keep iterating until interrupted.
