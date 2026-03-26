"""
Fixed harness for autoresearch-siren (SIREN V8 VQ-VAE). Agents must not edit this file.

- Verifies default paths (override with env vars).
- Exports TIME_BUDGET and evaluation helpers used by train.py.

Usage:
    uv run prepare.py
"""

from __future__ import annotations

import glob
import math
import os
import sys
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import amp as torch_amp
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify during agent experiments)
# ---------------------------------------------------------------------------

TIME_BUDGET = int(
    os.environ.get("SIREN_TIME_BUDGET", "300")
)  # wall-clock training seconds after warmup; override for smoke tests
WARMUP_TRAINING_STEPS = 10  # steps before accumulating wall time (compilation / first batches)

# Default paths relative to sibling SIREN checkout
_AUTORESEARCH_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIREN_ROOT = os.path.normpath(
    os.environ.get("SIREN_ROOT", os.path.join(_AUTORESEARCH_ROOT, "..", "SIREN"))
)

DEFAULT_DATA_DIR = os.environ.get(
    "SIREN_DATA_DIR", os.path.join(_SIREN_ROOT, "data", "waves_recovered_16k")
)
DEFAULT_VAL_DATA_DIR = os.environ.get("SIREN_VAL_DATA_DIR", "")
DEFAULT_CONFIG = os.environ.get(
    "SIREN_CONFIG",
    os.path.join(
        _SIREN_ROOT,
        "src",
        "ultra_low_bitrate_codec",
        "configs",
        "ultra58bps_16k.yaml",
    ),
)
DEFAULT_HUBERT_CKPT = os.environ.get(
    "SIREN_HUBERT_CKPT",
    os.path.join(_SIREN_ROOT, "checkpoints", "bithubert_distill", "bithubert_best.pt"),
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "SIREN_AUTORESEARCH_OUT",
    os.path.join(_AUTORESEARCH_ROOT, "checkpoints", "vqvae_autoresearch"),
)

VAL_FRACTION = float(os.environ.get("SIREN_VAL_FRACTION", "0.05"))
VAL_EVAL_MAX_BATCHES = int(os.environ.get("SIREN_VAL_EVAL_BATCHES", "32"))
STATS_BATCHES_EVAL = int(os.environ.get("SIREN_STATS_BATCHES_EVAL", "16"))

# Objective speech quality (literature — not SIREN-specific): ITU-T P.862 wideband PESQ,
# STOI (Taal et al.). Waveform is built from reconstructed HuBERT frames via a fixed
# linear mel projector + InverseMelScale + Griffin–Lim (torchaudio), so scores track
# training but are a proxy, not a full neural codec E2E test.
SKIP_SPEECH_METRICS = os.environ.get("SIREN_SKIP_SPEECH_METRICS", "0") == "1"
SPEECH_METRICS_MAX_CLIPS = int(os.environ.get("SIREN_SPEECH_METRICS_CLIPS", "12"))
GRIFFIN_LIM_ITER = int(os.environ.get("SIREN_GL_ITER", "24"))
SPEECH_SR = 16000


def siren_root() -> str:
    return _SIREN_ROOT


def list_wav_files(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.wav")))


def verify_assets(
    data_dir: str,
    config_path: str,
    hubert_ckpt: str,
    need_min_wavs: int = 1,
) -> None:
    """Exit with message if required files are missing."""
    if not os.path.isdir(data_dir):
        print(f"error: data_dir is not a directory: {data_dir}", file=sys.stderr)
        print("Set SIREN_DATA_DIR or place .wav files under SIREN data/.", file=sys.stderr)
        sys.exit(1)
    wavs = list_wav_files(data_dir)
    if len(wavs) < need_min_wavs:
        print(
            f"error: need at least {need_min_wavs} .wav in {data_dir}, found {len(wavs)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(hubert_ckpt):
        print(
            f"error: BitHuBERT checkpoint not found: {hubert_ckpt}",
            file=sys.stderr,
        )
        sys.exit(1)


def split_train_val_files(
    data_dir: str,
    val_fraction: float = VAL_FRACTION,
    val_data_dir: str = "",
) -> tuple[list[str], list[str]]:
    """Return (train_paths, val_paths). If val_data_dir is set, train=all in data_dir, val=all there."""
    if val_data_dir and os.path.isdir(val_data_dir):
        train = list_wav_files(data_dir)
        val = list_wav_files(val_data_dir)
        if not val:
            print(f"error: no .wav in val_data_dir {val_data_dir}", file=sys.stderr)
            sys.exit(1)
        return train, val
    all_files = list_wav_files(data_dir)
    n = len(all_files)
    n_val = max(1, int(n * val_fraction))
    if n <= 1:
        return all_files, all_files
    val_paths = all_files[-n_val:]
    train_paths = all_files[:-n_val]
    if not train_paths:
        train_paths = all_files
    return train_paths, val_paths


class WavListDataset(Dataset):
    """Load variable-length clips from explicit file paths (same cropping as train script)."""

    def __init__(self, paths: list[str], sample_rate: int = 16000, seed: int = 0):
        self.paths = paths
        self.sr = sample_rate
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import soundfile as sf

        wav, _ = sf.read(self.paths[idx])
        wav = torch.tensor(wav, dtype=torch.float32)
        max_len = 4 * self.sr
        if wav.shape[0] > max_len:
            start = torch.randint(
                0, wav.shape[0] - max_len, (1,), generator=self._gen
            ).item()
            wav = wav[start : start + max_len]
        return wav


def _collate_pad(batch: list[torch.Tensor]) -> torch.Tensor:
    """Pad to max length in batch (dim 0)."""
    max_len = max(int(x.shape[0]) for x in batch)
    out = torch.zeros(len(batch), max_len, dtype=batch[0].dtype)
    for i, x in enumerate(batch):
        out[i, : x.shape[0]] = x
    return out


def make_dataloader(
    paths: list[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    ds = WavListDataset(paths, seed=seed)
    kw: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_pad,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kw)


def _fixed_mel_proj_matrix(dtype: torch.dtype, dev: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    w = torch.randn(80, 768, generator=g, dtype=dtype) * (768**-0.5)
    return w.to(dev)


@torch.no_grad()
def measure_speech_pesq_stoi(
    val_paths: list[str],
    hubert: torch.nn.Module,
    fac: torch.nn.Module,
    rec: torch.nn.Module,
    sem_vq: torch.nn.Module,
    pro_vq: torch.nn.Module,
    spk_vq: torch.nn.Module,
    device: torch.device,
    hp: dict[str, Any],
    use_amp: bool,
    max_clips: int = SPEECH_METRICS_MAX_CLIPS,
) -> dict[str, float | int]:
    """
    Wideband PESQ (ITU-T P.862.2-style via ``pesq`` pkg) and STOI (Taal et al.).

    Reconstructed waveform is obtained with a **fixed, frozen** HuBERT→mel linear map
    and Griffin–Lim (torchaudio), so the score moves with ``h_recon`` quality. This is
    a **proxy** (not a full vocoder pipeline).
    """
    if SKIP_SPEECH_METRICS or not val_paths:
        return {
            "val_pesq_wb": float("nan"),
            "val_stoi": float("nan"),
            "speech_metrics_n": 0,
            "speech_metrics_ok": 0,
        }

    import numpy as np
    import soundfile as sf
    import torchaudio
    from pesq import pesq as pesq_fn
    from pystoi import stoi as stoi_fn

    amp_ctx = (
        torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
        if device.type == "cuda"
        else nullcontext()
    )

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=SPEECH_SR,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        power=2.0,
    ).to(device)
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=513, n_mels=80, sample_rate=SPEECH_SR
    ).to(device)
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=2.0,
        n_iter=GRIFFIN_LIM_ITER,
    ).to(device)

    W = _fixed_mel_proj_matrix(torch.float32, device)
    fsem = float(hp.get("fsq_sem_input_scale", hp.get("fsq_input_scale", 1.0)))
    fpro = float(hp.get("fsq_pro_input_scale", hp.get("fsq_input_scale", 1.0)))

    fac.eval()
    rec.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_vq.eval()

    pesq_vals: list[float] = []
    stoi_vals: list[float] = []

    for path in val_paths[: max(0, max_clips)]:
        try:
            wav_np, sr = sf.read(path, dtype="float32")
        except OSError:
            continue
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        if sr != SPEECH_SR:
            continue
        wav_t = torch.from_numpy(wav_np).float().to(device)
        max_len = 4 * SPEECH_SR
        if wav_t.shape[0] > max_len:
            start = (wav_t.shape[0] - max_len) // 2
            wav_t = wav_t[start : start + max_len]
        if wav_t.shape[0] < 2 * SPEECH_SR:
            continue
        wav_b = wav_t.unsqueeze(0).unsqueeze(1)

        with amp_ctx:
            h_feats, cnn_feats = hubert(wav_b)
            sem, pro, spk = fac(h_feats, cnn_feats)
            sem_z, _, _ = sem_vq(sem * fsem)
            pro_z, _, _ = pro_vq(pro * fpro)
            spk_z, _, _ = spk_vq(spk)
            h_recon = rec(sem_z, pro_z, spk_z, target_len=h_feats.shape[1])

        _no_amp = (
            torch_amp.autocast(device_type="cuda", enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )
        with _no_amp:
            h32 = h_recon.float()
            mel_ref = mel_tf(wav_b.squeeze(1)).clamp(min=1e-10)
            mel_pred = torch.matmul(h32, W.T).transpose(1, 2)
            mel_pred = torch.nn.functional.softplus(mel_pred).clamp(min=1e-10)
            t_ref = mel_ref.shape[-1]
            mel_pred = torch.nn.functional.interpolate(
                mel_pred,
                size=t_ref,
                mode="linear",
                align_corners=False,
            )
            mel_pred = mel_pred * (mel_ref.mean() / (mel_pred.mean() + 1e-8))
            lin = inv_mel(mel_pred)
            est = griffin(lin)
            ref_1d = wav_b.squeeze()
            est_1d = est.squeeze()[: ref_1d.shape[-1]]

        ref_np = ref_1d.detach().float().cpu().numpy().astype(np.float64)
        est_np = est_1d.detach().float().cpu().numpy().astype(np.float64)
        n = min(len(ref_np), len(est_np))
        if n < SPEECH_SR:
            continue
        ref_np = ref_np[:n]
        est_np = est_np[:n]
        ref_np = ref_np / (np.max(np.abs(ref_np)) + 1e-8)
        est_np = est_np / (np.max(np.abs(est_np)) + 1e-8)

        try:
            p = float(pesq_fn(SPEECH_SR, ref_np, est_np, "wb"))
            if math.isfinite(p):
                pesq_vals.append(p)
        except Exception:
            pass
        try:
            s = float(stoi_fn(ref_np, est_np, SPEECH_SR, extended=False))
            if math.isfinite(s):
                stoi_vals.append(s)
        except Exception:
            pass

    n_p, n_s = len(pesq_vals), len(stoi_vals)
    if n_p == 0 and n_s == 0:
        return {
            "val_pesq_wb": float("nan"),
            "val_stoi": float("nan"),
            "speech_metrics_n": 0,
            "speech_metrics_ok": 0,
        }

    return {
        "val_pesq_wb": float(np.nanmean(np.array(pesq_vals))) if pesq_vals else float("nan"),
        "val_stoi": float(np.nanmean(np.array(stoi_vals))) if stoi_vals else float("nan"),
        "speech_metrics_n": max(n_p, n_s),
        "speech_metrics_ok": 1,
    }


def compute_val_score(
    recon_mse: float,
    sem_h: float,
    pro_h: float,
    spk_h: float,
    sem_n: int,
    pro_n: int,
    spk_n: int,
) -> float:
    """
    Lower is better (like val_bpb). Primary term: reconstruction MSE.
    Large additive penalties discourage FSQ collapse.
    """
    score = float(recon_mse)
    if sem_h < 0.05 and sem_n <= 2:
        score += 10.0
    if pro_h < 0.05 and pro_n <= 2:
        score += 10.0
    if spk_h < 0.5 and spk_n <= 4:
        score += 5.0
    return score


@torch.no_grad()
def evaluate_vqvae_val(
    hubert: torch.nn.Module,
    fac: torch.nn.Module,
    rec: torch.nn.Module,
    sem_vq: torch.nn.Module,
    pro_vq: torch.nn.Module,
    spk_vq: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    hp: dict[str, Any],
    use_amp: bool,
    stats_batches: int = STATS_BATCHES_EVAL,
) -> dict[str, float | int]:
    """
    Mean recon MSE on val batches + code usage (H bits/tok, n_used) for Sem/Pro/Spk.
    """
    from ultra_low_bitrate_codec.utils.v8_residual_fsq import (
        semantic_index_stats,
        speaker_rfsq_indices_for_stats,
    )

    fac.eval()
    rec.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_vq.eval()

    fsem = float(hp.get("fsq_sem_input_scale", hp.get("fsq_input_scale", 1.0)))
    fpro = float(hp.get("fsq_pro_input_scale", hp.get("fsq_input_scale", 1.0)))

    total_recon = 0.0
    total_elems = 0
    sem_parts: list[torch.Tensor] = []
    pro_parts: list[torch.Tensor] = []
    spk_parts: list[torch.Tensor] = []

    amp_ctx = (
        torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
        if device.type == "cuda"
        else nullcontext()
    )

    it = iter(val_loader)
    for b in range(max(1, VAL_EVAL_MAX_BATCHES)):
        try:
            wav = next(it)
        except StopIteration:
            break
        wav = wav.to(device, non_blocking=True)
        with amp_ctx:
            h_feats, cnn_feats = hubert(wav.unsqueeze(1))
            sem, pro, spk = fac(h_feats, cnn_feats)
            sem_q = sem * fsem
            pro_q = pro * fpro
            sem_z, _, s_ix = sem_vq(sem_q)
            pro_z, _, p_ix = pro_vq(pro_q)
            spk_z, _, k_ix = spk_vq(spk)
            h_recon = rec(sem_z, pro_z, spk_z, target_len=h_feats.shape[1])
            if h_feats.shape[1] > h_recon.shape[1]:
                h_tgt = h_feats[:, : h_recon.shape[1], :]
            else:
                h_tgt = h_feats
            recon = F.mse_loss(h_recon, h_tgt)
        bs = int(wav.shape[0])
        total_recon += float(recon.item()) * bs
        total_elems += bs
        sem_parts.append(s_ix)
        pro_parts.append(p_ix)
        spk_parts.append(k_ix)

    mean_recon = total_recon / max(1, total_elems)

    # Optional richer histogram: more batches for index stats only
    it2 = iter(val_loader)
    sem_acc: list[torch.Tensor] = []
    pro_acc: list[torch.Tensor] = []
    spk_acc: list[torch.Tensor] = []
    for _ in range(max(1, stats_batches)):
        try:
            wav = next(it2)
        except StopIteration:
            it2 = iter(val_loader)
            wav = next(it2)
        wav = wav.to(device, non_blocking=True)
        h_s, c_s = hubert(wav.unsqueeze(1))
        sem_s, pro_s, spk_s = fac(h_s, c_s)
        sem_s = sem_s * fsem
        pro_s = pro_s * fpro
        _, _, s_ix = sem_vq(sem_s)
        _, _, p_ix = pro_vq(pro_s)
        _, _, k_ix = spk_vq(spk_s)
        sem_acc.append(s_ix)
        pro_acc.append(p_ix)
        spk_acc.append(k_ix)

    try:
        sem_idx = torch.cat(sem_acc, dim=0)
        pro_idx = torch.cat(pro_acc, dim=0)
        spk_idx = torch.cat(spk_acc, dim=0)
    except RuntimeError:
        sem_idx = sem_acc[-1]
        pro_idx = pro_acc[-1]
        spk_idx = spk_acc[-1]

    def _branch_stats(vq, indices: torch.Tensor, speaker: bool) -> tuple[float, int]:
        idx = speaker_rfsq_indices_for_stats(indices) if speaker else indices
        h_bits, _, n_used = semantic_index_stats(vq, idx)
        return float(h_bits), int(n_used)

    sem_h, sem_n = _branch_stats(sem_vq, sem_idx, False)
    pro_h, pro_n = _branch_stats(pro_vq, pro_idx, False)
    spk_h, spk_n = _branch_stats(spk_vq, spk_idx, True)

    val_score = compute_val_score(mean_recon, sem_h, pro_h, spk_h, sem_n, pro_n, spk_n)

    return {
        "val_recon_mse": mean_recon,
        "val_score": val_score,
        "sem_h_bits": sem_h,
        "pro_h_bits": pro_h,
        "spk_h_bits": spk_h,
        "sem_n_used": sem_n,
        "pro_n_used": pro_n,
        "spk_n_used": spk_n,
    }


if __name__ == "__main__":
    print(f"SIREN root: {_SIREN_ROOT}")
    print(f"Default DATA_DIR: {DEFAULT_DATA_DIR}")
    verify_assets(DEFAULT_DATA_DIR, DEFAULT_CONFIG, DEFAULT_HUBERT_CKPT)
    tr, va = split_train_val_files(
        DEFAULT_DATA_DIR, VAL_FRACTION, DEFAULT_VAL_DATA_DIR
    )
    print(f"Train wavs: {len(tr)}, Val wavs: {len(va)}")
    print("OK — run: uv run train.py")
