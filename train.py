"""
SIREN V8 VQ-VAE Phase 1 — autoresearch harness.
Time-budgeted training (see prepare.TIME_BUDGET). Agents edit this file only.

Usage: uv run train.py
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import amp as torch_amp
from tqdm import tqdm

from prepare import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_HUBERT_CKPT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAL_DATA_DIR,
    TIME_BUDGET,
    WARMUP_TRAINING_STEPS,
    evaluate_vqvae_val,
    make_dataloader,
    measure_speech_pesq_stoi,
    split_train_val_files,
    verify_assets,
)
from ultra_low_bitrate_codec.models.bithubert import BitHuBERT
from ultra_low_bitrate_codec.models.decoder import FeatureReconstructorV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.utils.v8_residual_fsq import (
    build_v8_residual_fsqs,
    semantic_index_stats,
    speaker_rfsq_indices_for_stats,
)

# ---------------------------------------------------------------------------
# LR schedule vs wall-clock progress (mirrors karpathy/autoresearch spirit)
# ---------------------------------------------------------------------------

WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.1
FINAL_LR_FRAC = 0.1


def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1.0 - cooldown) * FINAL_LR_FRAC


def _training_hparams(config: dict, args: argparse.Namespace) -> dict:
    tr = config.get("training", {})
    lr = (
        float(args.lr)
        if getattr(args, "lr", None) is not None
        else float(tr.get("learning_rate", 5e-5))
    )
    q_w = (
        float(args.quant_loss_weight)
        if getattr(args, "quant_loss_weight", None) is not None
        else float(tr.get("quant_loss_weight", 0.1))
    )
    return {
        "batch_size": int(tr.get("batch_size", 16)),
        "learning_rate": lr,
        "weight_decay": float(tr.get("weight_decay", 0.01)),
        "grad_clip": float(tr.get("grad_clip", 0.5)),
        "quant_loss_weight": q_w,
        "quant_loss_sem_weight": float(tr.get("quant_loss_sem_weight", 1.0)),
        "quant_loss_pro_weight": float(tr.get("quant_loss_pro_weight", 1.0)),
        "quant_loss_spk_weight": float(tr.get("quant_loss_spk_weight", 1.0)),
        "num_workers": int(tr.get("num_workers", 4)),
        "prefetch_factor": int(tr.get("prefetch_factor", 2)),
        "amp": bool(tr.get("amp", True)) and not getattr(args, "no_amp", False),
        "grad_accum_steps": max(1, int(tr.get("grad_accum_steps", 1))),
        "cudnn_benchmark": bool(tr.get("cudnn_benchmark", True)),
        "stats_batches": int(tr.get("stats_batches", 16)),
        "recon_weight": float(tr.get("recon_weight", 1.0)),
        "fsq_input_noise_std": float(tr.get("fsq_input_noise_std", 0.0)),
        "fsq_sem_noise_std": float(tr.get("fsq_sem_noise_std", 0.0)),
        "fsq_pro_noise_std": float(tr.get("fsq_pro_noise_std", 0.0)),
        "fsq_spk_noise_std": float(tr.get("fsq_spk_noise_std", 0.0)),
        "spread_weight": float(tr.get("spread_weight", 0.0)),
        "spread_floor": float(tr.get("spread_floor", 0.02)),
        "spk_spread_weight": float(tr.get("spk_spread_weight", 0.0)),
        "spk_spread_floor": tr.get("spk_spread_floor"),
        "fsq_input_scale": float(tr.get("fsq_input_scale", 1.0)),
        "fsq_sem_input_scale": float(
            tr.get("fsq_sem_input_scale", tr.get("fsq_input_scale", 1.0))
        ),
        "fsq_pro_input_scale": float(
            tr.get("fsq_pro_input_scale", tr.get("fsq_input_scale", 1.0))
        ),
        "latent_diversity_weight": float(tr.get("latent_diversity_weight", 0.0)),
        "latent_diversity_cap": float(tr.get("latent_diversity_cap", 0.5)),
        "latent_diversity_sem_weight": tr.get("latent_diversity_sem_weight"),
        "latent_diversity_pro_weight": tr.get("latent_diversity_pro_weight"),
        "latent_diversity_sem_cap": tr.get("latent_diversity_sem_cap"),
        "latent_diversity_pro_cap": tr.get("latent_diversity_pro_cap"),
        "batch_gram_logdet_eps": float(tr.get("batch_gram_logdet_eps", 1e-3)),
        "batch_gram_logdet_sem_weight": float(
            tr.get("batch_gram_logdet_sem_weight", 0.0)
        ),
        "batch_gram_logdet_pro_weight": float(
            tr.get("batch_gram_logdet_pro_weight", 0.0)
        ),
        "batch_gram_logdet_spk_weight": float(
            tr.get("batch_gram_logdet_spk_weight", 0.0)
        ),
    }


def _reinit_sem_pro_projection_heads(fac: InformationFactorizerV2) -> None:
    nn.init.normal_(fac.semantic_proj[2].weight, std=0.5)
    nn.init.zeros_(fac.semantic_proj[2].bias)
    nn.init.normal_(fac.prosody_proj[2].weight, std=0.5)
    nn.init.zeros_(fac.prosody_proj[2].bias)


def _latent_diversity_term(
    sem_pre: torch.Tensor,
    pro_pre: torch.Tensor,
    hp: dict,
) -> torch.Tensor:
    sem_bv = sem_pre.var(dim=0, unbiased=False).mean()
    pro_bv = pro_pre.var(dim=0, unbiased=False).mean()
    unified = float(hp.get("latent_diversity_weight", 0.0))
    lw_s = hp.get("latent_diversity_sem_weight")
    lw_p = hp.get("latent_diversity_pro_weight")
    if lw_s is not None or lw_p is not None:
        ldw_s = float(lw_s if lw_s is not None else unified)
        ldw_p = float(lw_p if lw_p is not None else unified)
        if hp.get("latent_diversity_sem_cap") is not None:
            scap = float(hp["latent_diversity_sem_cap"])
        else:
            scap = float(hp.get("latent_diversity_cap", 0.5))
        if hp.get("latent_diversity_pro_cap") is not None:
            pcap = float(hp["latent_diversity_pro_cap"])
        else:
            pcap = float(hp.get("latent_diversity_cap", 0.5))
        return ldw_s * sem_bv.clamp(max=scap) + ldw_p * pro_bv.clamp(max=pcap)
    if unified > 0:
        cap = float(hp.get("latent_diversity_cap", 0.5))
        return unified * (sem_bv + pro_bv).clamp(max=cap)
    return sem_bv * 0.0


def _batch_gram_logdet_term(z: torch.Tensor, eps: float) -> torch.Tensor:
    if z.dim() == 3:
        x = z.mean(dim=1)
    else:
        x = z
    b, d = int(x.shape[0]), int(x.shape[-1])
    if b < 2:
        return z.new_zeros(())
    x = x - x.mean(dim=0, keepdim=True)
    _no_amp = (
        torch_amp.autocast(device_type="cuda", enabled=False)
        if z.device.type == "cuda"
        else nullcontext()
    )
    with _no_amp:
        xf = x.float()
        g = (xf @ xf.T) / float(max(1, d))
        eye = torch.eye(b, device=z.device, dtype=torch.float32)
        g = g + float(eps) * eye
        jitter = float(eps)
        for _ in range(4):
            sign, logabsdet = torch.linalg.slogdet(g)
            if bool(torch.isfinite(logabsdet).item()) and sign.item() > 0:
                return -logabsdet
            jitter *= 4.0
            g = g + jitter * eye
    return z.new_zeros(())


def _spread_penalty_pre_fsq(x: torch.Tensor, floor: float) -> torch.Tensor:
    g = x.std()
    bv = x.var(dim=0, unbiased=False).mean().sqrt()
    return F.relu(floor - g) + F.relu(floor - bv)


def _log_branch_stats(
    vq,
    indices: torch.Tensor,
    label: str,
    speaker: bool = False,
) -> tuple[float, float, int]:
    idx = speaker_rfsq_indices_for_stats(indices) if speaker else indices
    h_bits, used_frac, n_used = semantic_index_stats(vq, idx)
    vocab = int(vq.vocab_size)
    max_bits = math.log2(vocab) if vocab > 1 else 0.0
    print(
        f"  {label}: H={h_bits:.2f} bits/tok (max {max_bits:.2f}), "
        f"used {n_used}/{vocab} ({100.0 * used_frac:.1f}%)",
        flush=True,
    )
    return h_bits, used_frac, n_used


@torch.no_grad()
def _collect_indices_multi_batch(
    dataloader,
    hubert: torch.nn.Module,
    fac: torch.nn.Module,
    sem_vq: torch.nn.Module,
    pro_vq: torch.nn.Module,
    spk_vq: torch.nn.Module,
    device: torch.device,
    num_batches: int,
    fsq_sem_input_scale: float = 1.0,
    fsq_pro_input_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sem_parts: list[torch.Tensor] = []
    pro_parts: list[torch.Tensor] = []
    spk_parts: list[torch.Tensor] = []
    it = iter(dataloader)
    for _ in range(max(1, num_batches)):
        try:
            wav = next(it)
        except StopIteration:
            it = iter(dataloader)
            wav = next(it)
        wav = wav.to(device, non_blocking=True)
        h_s, c_s = hubert(wav.unsqueeze(1))
        sem_s, pro_s, spk_s = fac(h_s, c_s)
        if fsq_sem_input_scale != 1.0:
            sem_s = sem_s * fsq_sem_input_scale
        if fsq_pro_input_scale != 1.0:
            pro_s = pro_s * fsq_pro_input_scale
        _, _, s_ix = sem_vq(sem_s)
        _, _, p_ix = pro_vq(pro_s)
        _, _, k_ix = spk_vq(spk_s)
        sem_parts.append(s_ix)
        pro_parts.append(p_ix)
        spk_parts.append(k_ix)
    try:
        return (
            torch.cat(sem_parts, dim=0),
            torch.cat(pro_parts, dim=0),
            torch.cat(spk_parts, dim=0),
        )
    except RuntimeError:
        return sem_parts[-1], pro_parts[-1], spk_parts[-1]


def _set_lr(optimizer: torch.optim.Optimizer, base_lr: float, progress: float) -> None:
    m = get_lr_multiplier(progress)
    for g in optimizer.param_groups:
        g["lr"] = base_lr * m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIREN V8 VQ-VAE (autoresearch time budget)"
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--resume_ckpt", default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--quant_loss_weight", type=float, default=None)
    parser.add_argument("--quant_sem_weight", type=float, default=None)
    parser.add_argument("--quant_pro_weight", type=float, default=None)
    parser.add_argument("--quant_spk_weight", type=float, default=None)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--stats_batches", type=int, default=None)
    parser.add_argument("--recon_weight", type=float, default=None)
    parser.add_argument("--reset_sem_pro_fsq", action="store_true")
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Logical epoch label for reset_sem_pro_fsq scheduler fast-forward only.",
    )
    args = parser.parse_args()

    if args.reset_sem_pro_fsq and (
        not args.resume_ckpt or not os.path.exists(args.resume_ckpt)
    ):
        raise SystemExit(
            "error: --reset_sem_pro_fsq requires --resume_ckpt to an existing checkpoint"
        )

    data_dir = os.path.abspath(args.data_dir)
    config_path = os.path.abspath(args.config)
    hubert_ckpt = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, config_path, hubert_ckpt)

    val_dir = args.val_data_dir.strip() if args.val_data_dir else ""
    train_paths, val_paths = split_train_val_files(
        data_dir,
        val_data_dir=val_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hp = _training_hparams(config, args)
    if args.grad_accum is not None:
        hp["grad_accum_steps"] = max(1, int(args.grad_accum))
    if args.stats_batches is not None:
        hp["stats_batches"] = max(1, int(args.stats_batches))
    if args.quant_sem_weight is not None:
        hp["quant_loss_sem_weight"] = float(args.quant_sem_weight)
    if args.quant_pro_weight is not None:
        hp["quant_loss_pro_weight"] = float(args.quant_pro_weight)
    if args.quant_spk_weight is not None:
        hp["quant_loss_spk_weight"] = float(args.quant_spk_weight)
    if args.recon_weight is not None:
        hp["recon_weight"] = float(args.recon_weight)

    if device.type == "cuda" and hp["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True

    pin = device.type == "cuda"
    train_loader = make_dataloader(
        train_paths,
        hp["batch_size"],
        shuffle=True,
        num_workers=hp["num_workers"],
        prefetch_factor=hp["prefetch_factor"],
        pin_memory=pin,
        seed=int(time.time()) % (2**31),
    )
    val_loader = make_dataloader(
        val_paths,
        hp["batch_size"],
        shuffle=False,
        num_workers=min(2, hp["num_workers"]),
        prefetch_factor=hp["prefetch_factor"],
        pin_memory=pin,
        seed=0,
    )

    hubert = BitHuBERT(hidden_dim=384, output_dim=768, num_layers=12).to(device).eval()
    hubert.load_state_dict(torch.load(hubert_ckpt, map_location=device))

    fac = InformationFactorizerV2(config).to(device)
    rec = FeatureReconstructorV2(config).to(device)
    sem_vq, pro_vq, spk_vq = build_v8_residual_fsqs(config, device)

    params = (
        list(fac.parameters())
        + list(rec.parameters())
        + list(sem_vq.parameters())
        + list(pro_vq.parameters())
        + list(spk_vq.parameters())
    )
    base_lr = float(hp["learning_rate"])
    optimizer = torch.optim.AdamW(
        params,
        lr=base_lr,
        weight_decay=hp["weight_decay"],
    )
    for g in optimizer.param_groups:
        g["initial_lr"] = base_lr

    steps_per_epoch = max(
        1, math.ceil(len(train_loader) / hp["grad_accum_steps"])
    )
    use_amp = hp["amp"] and device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)
    global_step = 0

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        fac.load_state_dict(ckpt["factorizer"])
        rec.load_state_dict(ckpt["reconstructor"])
        spk_vq.load_state_dict(ckpt["spk_vq"])
        if args.reset_sem_pro_fsq:
            _reinit_sem_pro_projection_heads(fac)
            global_step = int(args.start_epoch * steps_per_epoch)
            print(
                "Loaded partial ckpt (--reset_sem_pro_fsq); new sem_vq/pro_vq; "
                f"global_step={global_step}"
            )
        else:
            sem_vq.load_state_dict(ckpt["sem_vq"])
            pro_vq.load_state_dict(ckpt["pro_vq"])
            global_step = int(
                ckpt.get("global_step", args.start_epoch * steps_per_epoch)
            )
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler"])

    os.makedirs(args.output_dir, exist_ok=True)

    print("SIREN V8 VQ-VAE (autoresearch)")
    print(f"  config: {config_path}")
    print(f"  TIME_BUDGET: {TIME_BUDGET}s (wall after step {WARMUP_TRAINING_STEPS})")
    print(f"  train wavs: {len(train_paths)}, val wavs: {len(val_paths)}")
    print(f"  device: {device}, amp={use_amp}, accum={hp['grad_accum_steps']}")

    t_start = time.time()
    total_training_time = 0.0
    optim_step = 0
    optimizer.zero_grad(set_to_none=True)
    smooth_loss = 0.0
    done = False
    epoch_id = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_iter = iter(train_loader)

    def _next_wav() -> torch.Tensor:
        nonlocal train_iter, epoch_id
        try:
            return next(train_iter)
        except StopIteration:
            epoch_id += 1
            train_iter = iter(train_loader)
            return next(train_iter)

    while not done:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        fac.train()
        rec.train()
        sem_vq.train()
        pro_vq.train()
        spk_vq.train()

        last_raw = 0.0
        for _micro in range(hp["grad_accum_steps"]):
            wav = _next_wav()
            wav = wav.to(device, non_blocking=True)
            with torch.no_grad():
                h_feats, cnn_feats = hubert(wav.unsqueeze(1))

            amp_ctx = (
                torch_amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                )
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                sem, pro, spk = fac(h_feats, cnn_feats)
                sem_pre, pro_pre = sem, pro
                fsem = float(
                    hp.get("fsq_sem_input_scale", hp.get("fsq_input_scale", 1.0))
                )
                fpro = float(
                    hp.get("fsq_pro_input_scale", hp.get("fsq_input_scale", 1.0))
                )
                sem = sem_pre * fsem
                pro = pro_pre * fpro
                nstd = float(hp.get("fsq_input_noise_std", 0.0))
                if fac.training and nstd > 0:
                    sem = sem + nstd * torch.randn_like(sem)
                    pro = pro + nstd * torch.randn_like(pro)
                snstd = float(hp.get("fsq_sem_noise_std", 0.0))
                if fac.training and snstd > 0:
                    sem = sem + snstd * torch.randn_like(sem)
                pnstd = float(hp.get("fsq_pro_noise_std", 0.0))
                if fac.training and pnstd > 0:
                    pro = pro + pnstd * torch.randn_like(pro)
                sem_z, sem_q_loss, _ = sem_vq(sem)
                pro_z, pro_q_loss, _ = pro_vq(pro)
                spk_in = spk
                spk_nstd = float(hp.get("fsq_spk_noise_std", 0.0))
                if fac.training and spk_nstd > 0:
                    spk_in = spk_in + spk_nstd * torch.randn_like(spk_in)
                spk_z, spk_q_loss, _ = spk_vq(spk_in)
                h_recon = rec(sem_z, pro_z, spk_z, target_len=h_feats.shape[1])
                h_tgt = h_feats
                if h_tgt.shape[1] > h_recon.shape[1]:
                    h_tgt = h_tgt[:, : h_recon.shape[1], :]
                recon = F.mse_loss(h_recon, h_tgt)
                q = (
                    hp["quant_loss_sem_weight"] * sem_q_loss
                    + hp["quant_loss_pro_weight"] * pro_q_loss
                    + hp["quant_loss_spk_weight"] * spk_q_loss
                )
                raw_loss = hp["recon_weight"] * recon + hp["quant_loss_weight"] * q
                # exp11: apply spread penalty to ALL branches (including speaker) to prevent collapse
                sw = hp.get("spread_weight", 0.0)
                if sw > 0:
                    fl = hp.get("spread_floor", 0.02)
                    raw_loss = raw_loss + sw * (
                        _spread_penalty_pre_fsq(sem_pre, fl)
                        + _spread_penalty_pre_fsq(pro_pre, fl)
                        + _spread_penalty_pre_fsq(spk.unsqueeze(1), fl)
                    )
                # exp11: REMOVED broken diversity term (was subtracted, encouraging LOW variance = collapse)
                # _latent_diversity_term removed
                bge = float(hp.get("batch_gram_logdet_eps", 1e-3))
                w_bg_sem = float(hp.get("batch_gram_logdet_sem_weight", 0.0))
                if w_bg_sem > 0:
                    raw_loss = raw_loss + w_bg_sem * _batch_gram_logdet_term(
                        sem_pre, bge
                    )
                w_bg_pro = float(hp.get("batch_gram_logdet_pro_weight", 0.0))
                if w_bg_pro > 0:
                    raw_loss = raw_loss + w_bg_pro * _batch_gram_logdet_term(
                        pro_pre, bge
                    )
                w_bg_spk = float(hp.get("batch_gram_logdet_spk_weight", 0.0))
                if w_bg_spk > 0:
                    raw_loss = raw_loss + w_bg_spk * _batch_gram_logdet_term(spk, bge)
                loss_scaled = raw_loss / hp["grad_accum_steps"]

            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            last_raw = float(raw_loss.detach().item())

        train_loss_f = last_raw
        if math.isnan(train_loss_f) or train_loss_f > 1e4:
            print("FAIL", flush=True)
            raise SystemExit(1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if optim_step > WARMUP_TRAINING_STEPS:
            total_training_time += dt

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        _set_lr(optimizer, base_lr, progress)

        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        global_step += 1
        optim_step += 1
        optimizer.zero_grad(set_to_none=True)

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss_f
        deb = smooth_loss / (1.0 - ema_beta ** min(optim_step, 10**9))
        rem = max(0.0, TIME_BUDGET - total_training_time)
        print(
            f"\rstep {optim_step:05d} ({100 * progress:.1f}%) | loss: {deb:.4f} | "
            f"dt: {dt * 1000:.0f}ms | epoch: {epoch_id} | rem: {rem:.0f}s ",
            end="",
            flush=True,
        )

        if optim_step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()

        if optim_step > WARMUP_TRAINING_STEPS and total_training_time >= TIME_BUDGET:
            done = True

    print(flush=True)

    # Train code stats (same as epoch-end in scripts/train_v8_vqvae.py)
    fac.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_vq.eval()
    with torch.no_grad():
        sem_idx, pro_idx, spk_idx = _collect_indices_multi_batch(
            train_loader,
            hubert,
            fac,
            sem_vq,
            pro_vq,
            spk_vq,
            device,
            hp["stats_batches"],
            fsq_sem_input_scale=float(hp.get("fsq_sem_input_scale", 1.0)),
            fsq_pro_input_scale=float(hp.get("fsq_pro_input_scale", 1.0)),
        )
    print(
        f"  Code stats (train, ~{hp['stats_batches']} batches):",
        flush=True,
    )
    _log_branch_stats(sem_vq, sem_idx, "Sem")
    _log_branch_stats(pro_vq, pro_idx, "Pro")
    _log_branch_stats(spk_vq, spk_idx, "Spk", speaker=True)

    metrics = evaluate_vqvae_val(
        hubert,
        fac,
        rec,
        sem_vq,
        pro_vq,
        spk_vq,
        val_loader,
        device,
        hp,
        use_amp,
        stats_batches=hp["stats_batches"],
    )
    speech = measure_speech_pesq_stoi(
        val_paths,
        hubert,
        fac,
        rec,
        sem_vq,
        pro_vq,
        spk_vq,
        device,
        hp,
        use_amp,
    )

    ckpt_payload = {
        "factorizer": fac.state_dict(),
        "reconstructor": rec.state_dict(),
        "sem_vq": sem_vq.state_dict(),
        "pro_vq": pro_vq.state_dict(),
        "spk_vq": spk_vq.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "config_path": config_path,
        "epoch": epoch_id,
    }
    if use_amp:
        ckpt_payload["scaler"] = scaler.state_dict()
    out_pt = os.path.join(args.output_dir, "vqvae_autoresearch_last.pt")
    torch.save(ckpt_payload, out_pt)

    t_end = time.time()
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_score:          {float(metrics['val_score']):.6f}")
    print(f"val_recon_mse:      {float(metrics['val_recon_mse']):.6f}")
    def _fmt_metric(x: object) -> str:
        if isinstance(x, float) and math.isfinite(x):
            return f"{x:.6f}"
        return "nan"

    print(f"val_pesq_wb:        {_fmt_metric(speech['val_pesq_wb'])}")
    print(f"val_stoi:           {_fmt_metric(speech['val_stoi'])}")
    print(f"speech_metrics_n:   {int(speech['speech_metrics_n'])}")
    print(f"sem_h_bits:         {float(metrics['sem_h_bits']):.4f}")
    print(f"pro_h_bits:         {float(metrics['pro_h_bits']):.4f}")
    print(f"spk_h_bits:         {float(metrics['spk_h_bits']):.4f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {optim_step}")
    print(f"checkpoint:         {out_pt}")


if __name__ == "__main__":
    main()
