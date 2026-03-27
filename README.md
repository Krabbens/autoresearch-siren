# autoresearch-siren

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for **SIREN V8 VQ-VAE Phase 1** (BitHuBERT → factorizer → Residual FSQ → feature reconstructor). The agent-editable surface is **`train.py`**; **`prepare.py`** holds the time budget, path defaults, and fixed validation helpers.

**Agent policy:** SIREN is **not** treated as canonical “good” code — see **`program.md` → “SIREN code is not sacred”**: you may rewrite architecture and pipeline **entirely inside `train.py`**, validate with `uv run train.py`, and **log every run to `results.tsv`**.

## Layout

- Sibling checkout: **`../SIREN`** (editable dependency `siren-codec` in `pyproject.toml`).
- Default data: `../SIREN/data/waves_recovered_16k/*.wav`
- Default config: `../SIREN/src/ultra_low_bitrate_codec/configs/ultra58bps_16k.yaml`
- Default HuBERT: `../SIREN/checkpoints/bithubert_distill/bithubert_best.pt`

Override paths with env vars documented in `prepare.py` (`SIREN_ROOT`, `SIREN_DATA_DIR`, `SIREN_CONFIG`, `SIREN_HUBERT_CKPT`, `SIREN_AUTORESEARCH_OUT`, …).

## Requirements

- **Python ≥ 3.11** (transitive deps e.g. `onnxruntime` need 3.11+ wheels).
- [uv](https://docs.astral.sh/uv/)
- One **CUDA** GPU recommended (same as training SIREN).

## New machine (clone + data + deps)

Use **HTTPS** if you do not have GitHub SSH keys on that machine (`git@github.com:...` fails with “Permission denied (publickey)”).

```bash
git clone https://github.com/Krabbens/autoresearch-siren.git
cd autoresearch-siren
git checkout autoresearch/mar27
```

Install [GitHub CLI](https://cli.github.com/) (`gh`), then run `gh auth login` once. After that:

```bash
./setup.sh
```

`setup.sh` clones **SIREN** next to this repo, downloads the training tarball from [Releases](https://github.com/Krabbens/autoresearch-siren/releases), extracts checkpoints, and creates a venv. To clone SIREN via SSH instead: `USE_SSH_CLONE=1 ./setup.sh`.

## Quick start

```bash
cd autoresearch-siren
uv sync
uv run prepare.py    # verify wavs, config, checkpoint
uv run train.py      # trains until prepare.TIME_BUDGET (default 300s wall after warmup)
```

Smoke test (short budget):

```bash
SIREN_TIME_BUDGET=30 uv run train.py
```

## Outputs

- Training log ends with `---` and **`val_score:`** (lower is better), plus literature **speech** metrics **`val_pesq_wb`** (wideband PESQ) and **`val_stoi`** (higher is better). See `prepare.py` for the Griffin–Lim proxy path. Env **`SIREN_SKIP_SPEECH_METRICS=1`** skips them for faster iteration.
- See `program.md` for the autonomous research loop and `results.tsv` format.
- Last weights: `checkpoints/vqvae_autoresearch/vqvae_autoresearch_last.pt`.

## Upstream artifacts

Files such as `analysis.ipynb` / `progress.png` are from the original autoresearch repo and are optional for this harness.
