"""
SIREN VQ-VAE Phase 1 — Improved architecture with separate branches.

Building on exp14's success, this version:
1. Re-introduces sem/pro/spk branches (but with working VQ, not broken FSQ)
2. Proper speech metrics via prepare.py helpers
3. Larger latent dimensions for better reconstruction
4. Full validation set evaluation

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


# ---------------------------------------------------------------------------
# Vector Quantizer with codebook reset
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Vector Quantizer with commitment loss and codebook reset."""
    
    def __init__(self, num_codes, codebook_dim, beta=0.25, reset_threshold=0.001):
        super().__init__()
        self.num_codes = num_codes
        self.codebook_dim = codebook_dim
        self.beta = beta  # Commitment loss weight
        self.reset_threshold = reset_threshold
        
        self.embedding = nn.Embedding(num_codes, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        self.register_buffer("ema_count", torch.ones(num_codes))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())
        
    def forward(self, z):
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)
        
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) 
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).reshape(z.shape)
        
        # Commitment loss: encoder should output vectors close to codebook
        commit_loss = F.mse_loss(z_q.detach(), z) * self.beta
        # Codebook loss: codebook should move towards encoder outputs
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = commit_loss + codebook_loss
        
        z_q = z + (z_q - z).detach()
        
        if self.training:
            self._update_usage(indices)
        
        entropy_bits = self._compute_entropy(indices)
        return z_q, vq_loss, indices, entropy_bits
    
    def _update_usage(self, indices):
        counts = torch.bincount(indices.flatten(), minlength=self.num_codes)
        self.ema_count = 0.99 * self.ema_count + 0.01 * counts.float()
        
    def _compute_entropy(self, indices):
        counts = torch.bincount(indices.flatten(), minlength=self.num_codes).float()
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 0]
        return -(probs * torch.log2(probs)).sum().item()
    
    def reset_dead_codes(self, z):
        if self.ema_count.min() < self.reset_threshold:
            dead_mask = self.ema_count < self.reset_threshold
            num_dead = dead_mask.sum().item()
            z_flat = z.reshape(-1, self.codebook_dim).float()
            rand_idx = torch.randint(0, len(z_flat), (num_dead,), device=z.device)
            with torch.no_grad():
                self.embedding.weight[dead_mask] = z_flat[rand_idx].to(self.embedding.weight.dtype)
                self.ema_count[dead_mask] = 1.0


# ---------------------------------------------------------------------------
# Encoder and Decoder with separate branches
# ---------------------------------------------------------------------------

class BranchEncoder(nn.Module):
    """Encoder with separate semantic, prosody, speaker branches."""
    
    def __init__(self, input_dim=768, hidden_dim=512, 
                 sem_dim=128, pro_dim=64, spk_dim=256,
                 pro_temporal_compression=1):
        super().__init__()
        self.input_dim = input_dim
        self.sem_dim = sem_dim
        self.pro_dim = pro_dim
        self.spk_dim = spk_dim
        self.pro_temporal_compression = pro_temporal_compression
        
        # Shared encoder for sem/pro
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Semantic branch (temporal, fine-grained)
        self.semantic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, sem_dim),
        )
        
        # Prosody branch (temporal compression for efficiency)
        self.prosody = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, pro_dim),
        )
        
        # Speaker branch - TEMPORAL (not global pooling!)
        self.speaker_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, spk_dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, 768) HuBERT features
        Returns:
            sem: (B, T, sem_dim) semantic features
            pro: (B, T', pro_dim) prosody features (T' = T // compression)
            spk: (B, T, spk_dim) temporal speaker features
        """
        # Shared path for sem/pro
        h = self.shared(x)
        sem = self.semantic(h)  # (B, T, sem_dim)
        
        # Prosody with temporal compression
        pro_full = self.prosody(h)  # (B, T, pro_dim)
        if self.pro_temporal_compression > 1:
            # Downsample by averaging frames
            B, T, D = pro_full.shape
            T_compressed = T // self.pro_temporal_compression
            pro = pro_full[:, :T_compressed * self.pro_temporal_compression, :]
            pro = pro.reshape(B, T_compressed, self.pro_temporal_compression, D)
            pro = pro.mean(dim=2)  # (B, T', pro_dim)
        else:
            pro = pro_full
        
        # Speaker: temporal encoding (no pooling here)
        spk = self.speaker_encoder(x)  # (B, T, spk_dim)
        
        return sem, pro, spk


class BranchDecoder(nn.Module):
    """Decoder that reconstructs from quantized sem/pro/spk."""
    
    def __init__(self, sem_dim=128, pro_dim=64, spk_dim=256, 
                 hidden_dim=512, output_dim=768,
                 pro_temporal_compression=1):
        super().__init__()
        self.pro_temporal_compression = pro_temporal_compression
        total_dim = sem_dim + pro_dim + spk_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, sem_q, pro_q, spk_q, target_len=None):
        """
        Args:
            sem_q: (B, T, sem_dim) quantized semantic
            pro_q: (B, T', pro_dim) quantized prosody (T' may be < T)
            spk_q: (B, T, spk_dim) quantized speaker
            target_len: optional, ignored (we use sem_q's T)
        Returns:
            x_recon: (B, T, 768) reconstructed features
        """
        B, T, _ = sem_q.shape
        
        # Upsample prosody if compressed
        if self.pro_temporal_compression > 1 and pro_q.shape[1] < T:
            # Simple nearest neighbor upsampling
            pro_q = pro_q.transpose(1, 2)  # (B, D, T')
            pro_q = F.interpolate(pro_q, size=T, mode='nearest')  # (B, D, T)
            pro_q = pro_q.transpose(1, 2)  # (B, T, D)
        
        # Handle different speaker shapes
        if spk_q.dim() == 2:  # (B, spk_dim) global
            spk_expanded = spk_q.unsqueeze(1).expand(-1, T, -1)
        elif spk_q.dim() == 4:  # (B, 1, T, spk_dim) from prepare.py
            spk_expanded = spk_q.squeeze(1)  # (B, T, spk_dim)
        else:  # (B, T, spk_dim) temporal
            spk_expanded = spk_q
        
        # Concatenate and decode
        combined = torch.cat([sem_q, pro_q, spk_expanded], dim=-1)
        return self.net(combined)


# ---------------------------------------------------------------------------
# Entropy Regularization
# ---------------------------------------------------------------------------

def entropy_bonus(indices, num_codes, target_entropy=None, weight=0.1):
    """Add bonus for high entropy code usage."""
    counts = torch.bincount(indices.flatten(), minlength=num_codes).float()
    probs = counts / (counts.sum() + 1e-10)
    probs_nonzero = probs[probs > 0]
    current_entropy = -(probs_nonzero * torch.log2(probs_nonzero)).sum()
    
    if target_entropy is None:
        target_entropy = 0.8 * math.log2(num_codes)
    
    if current_entropy < target_entropy:
        return (target_entropy - current_entropy) * weight
    return current_entropy * 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress: float) -> float:
    WARMUP_RATIO = 0.05
    WARMDOWN_RATIO = 0.1
    FINAL_LR_FRAC = 0.1
    
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1.0 - cooldown) * FINAL_LR_FRAC


def _set_lr(optimizer, base_lr, progress):
    m = get_lr_multiplier(progress)
    for g in optimizer.param_groups:
        g["lr"] = base_lr * m


class SimpleFactorizer(nn.Module):
    """Wrapper to make BranchEncoder compatible with prepare.py eval."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, h, c):
        sem, pro, spk = self.encoder(h)
        # prepare.py expects spk to be (B, 1, T, spk_dim) for some reason
        # Return as (B, T, spk_dim) which decoder handles
        return sem, pro, spk


def main():
    parser = argparse.ArgumentParser(description="Improved VQ-VAE with branches")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--resume_ckpt", default=None)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    
    # Architecture hyperparameters
    parser.add_argument("--sem_dim", type=int, default=128)
    parser.add_argument("--pro_dim", type=int, default=64)
    parser.add_argument("--spk_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    
    # Temporal compression (exp19: prosody at lower frame rate)
    parser.add_argument("--pro_temporal_compression", type=int, default=1)  # 1=no compression, 2=2x, 4=4x, etc.
    
    # VQ hyperparameters
    parser.add_argument("--num_codes", type=int, default=1024)  # Larger codebook
    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--commitment_beta", type=float, default=16.0)  # exp28: test very strong commitment
    parser.add_argument("--entropy_weight", type=float, default=1.0)  # Higher for all
    parser.add_argument("--entropy_weight_spk", type=float, default=5.0)  # Much higher for speaker
    parser.add_argument("--reset_threshold", type=float, default=0.0005)  # Reset dead codes sooner
    
    args = parser.parse_args()

    # Setup paths
    data_dir = os.path.abspath(args.data_dir)
    config_path = os.path.abspath(args.config)
    hubert_ckpt = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, config_path, hubert_ckpt)

    val_dir = args.val_data_dir.strip() if args.val_data_dir else ""
    train_paths, val_paths = split_train_val_files(data_dir, val_data_dir=val_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training hyperparameters
    batch_size = 32
    base_lr = args.lr
    weight_decay = 0.01
    grad_clip = 0.5
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Data loaders
    pin = device.type == "cuda"
    train_loader = make_dataloader(
        train_paths, batch_size, shuffle=True, num_workers=4,
        prefetch_factor=2, pin_memory=pin, seed=int(time.time()) % (2**31)
    )
    val_loader = make_dataloader(
        val_paths, batch_size, shuffle=False, num_workers=2,
        prefetch_factor=2, pin_memory=pin, seed=0
    )

    # Models
    hubert = BitHuBERT(hidden_dim=384, output_dim=768, num_layers=12).to(device).eval()
    hubert.load_state_dict(torch.load(hubert_ckpt, map_location=device))

    encoder = BranchEncoder(
        input_dim=768, hidden_dim=args.hidden_dim,
        sem_dim=args.sem_dim, pro_dim=args.pro_dim, spk_dim=args.spk_dim,
        pro_temporal_compression=args.pro_temporal_compression
    ).to(device)
    
    decoder = BranchDecoder(
        sem_dim=args.sem_dim, pro_dim=args.pro_dim, spk_dim=args.spk_dim,
        hidden_dim=args.hidden_dim, output_dim=768,
        pro_temporal_compression=args.pro_temporal_compression
    ).to(device)
    
    # Separate VQ for each branch
    sem_vq = VectorQuantizer(num_codes=args.num_codes, codebook_dim=args.sem_dim,
                             beta=args.commitment_beta, reset_threshold=args.reset_threshold).to(device)
    pro_vq = VectorQuantizer(num_codes=args.num_codes, codebook_dim=args.pro_dim,
                             beta=args.commitment_beta, reset_threshold=args.reset_threshold).to(device)
    spk_vq = VectorQuantizer(num_codes=args.num_codes, codebook_dim=args.spk_dim,
                             beta=args.commitment_beta, reset_threshold=args.reset_threshold).to(device)

    # Optimizer
    params = (list(encoder.parameters()) + list(decoder.parameters()) + 
              list(sem_vq.parameters()) + list(pro_vq.parameters()) + list(spk_vq.parameters()))
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    
    # AMP
    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    global_step = 0
    optim_step = 0
    
    print("Improved VQ-VAE with branches (autoresearch)")
    print(f"  sem_dim={args.sem_dim}, pro_dim={args.pro_dim}, spk_dim={args.spk_dim}")
    print(f"  num_codes={args.num_codes}, entropy_weight={args.entropy_weight}")
    print(f"  TIME_BUDGET: {TIME_BUDGET}s (wall after step {WARMUP_TRAINING_STEPS})")
    print(f"  train wavs: {len(train_paths)}, val wavs: {len(val_paths)}")
    print(f"  device: {device}, amp={use_amp}")

    t_start = time.time()
    total_training_time = 0.0
    smooth_loss = 0.0
    done = False
    epoch_id = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_iter = iter(train_loader)

    def _next_wav():
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

        encoder.train()
        decoder.train()
        sem_vq.train()
        pro_vq.train()
        spk_vq.train()

        wav = _next_wav()
        wav = wav.to(device, non_blocking=True)
        
        with torch.no_grad():
            h_feats, _ = hubert(wav.unsqueeze(1))

        amp_ctx = (
            torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            if device.type == "cuda"
            else nullcontext()
        )
        
        with amp_ctx:
            # Encode
            sem, pro, spk = encoder(h_feats)
            
            # Quantize each branch
            sem_q, sem_vq_loss, sem_idx, sem_entropy = sem_vq(sem)
            pro_q, pro_vq_loss, pro_idx, pro_entropy = pro_vq(pro)
            # Speaker is now temporal (B, T, D) - quantize per timestep
            spk_q, spk_vq_loss, spk_idx, spk_entropy = spk_vq(spk)
            
            # Decode
            h_recon = decoder(sem_q, pro_q, spk_q)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(h_recon, h_feats)
            
            # Total VQ loss
            total_vq_loss = sem_vq_loss + pro_vq_loss + spk_vq_loss
            
            # Entropy bonuses (encourage code usage in each branch)
            # Speaker gets higher weight since it tends to collapse
            sem_ent_bonus = entropy_bonus(sem_idx, args.num_codes, weight=args.entropy_weight)
            pro_ent_bonus = entropy_bonus(pro_idx, args.num_codes, weight=args.entropy_weight)
            spk_ent_bonus = entropy_bonus(spk_idx, args.num_codes, weight=args.entropy_weight_spk)
            
            # Total loss
            raw_loss = recon_loss + args.vq_weight * total_vq_loss + sem_ent_bonus + pro_ent_bonus + spk_ent_bonus

        loss_scaled = raw_loss
        
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        train_loss_f = float(raw_loss.detach().item())
        
        if math.isnan(train_loss_f) or train_loss_f > 1e4:
            print("FAIL - NaN or explosion", flush=True)
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
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Periodic codebook reset
        if global_step % 100 == 0:
            sem_vq.reset_dead_codes(sem)
            pro_vq.reset_dead_codes(pro)
            spk_vq.reset_dead_codes(spk)  # Now temporal, no unsqueeze needed
        
        global_step += 1
        optim_step += 1
        optimizer.zero_grad(set_to_none=True)

        # EMA smoothing
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss_f
        deb = smooth_loss / (1.0 - ema_beta ** min(optim_step, 10**9))
        rem = max(0.0, TIME_BUDGET - total_training_time)
        
        avg_entropy = (sem_entropy + pro_entropy + spk_entropy) / 3
        
        print(
            f"\rstep {optim_step:05d} ({100 * progress:.1f}%) | "
            f"loss: {deb:.4f} | recon: {recon_loss.item():.4f} | "
            f"vq: {total_vq_loss.item():.4f} | H: {avg_entropy:.2f} | "
            f"dt: {dt * 1000:.0f}ms | rem: {rem:.0f}s ",
            end="", flush=True
        )

        if optim_step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()

        if optim_step > WARMUP_TRAINING_STEPS and total_training_time >= TIME_BUDGET:
            done = True

    print(flush=True)

    # Code usage stats per branch
    sem_vq.eval()
    pro_vq.eval()
    spk_vq.eval()
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        sem_all, pro_all, spk_all = [], [], []
        for batch_idx, wav_path in enumerate(train_paths[:batch_size * 4]):
            import soundfile as sf
            wav_np, _ = sf.read(wav_path)
            wav_t = torch.tensor(wav_np, dtype=torch.float32).to(device)
            if wav_t.shape[0] > 4 * 16000:
                wav_t = wav_t[:4 * 16000]
            wav_b = wav_t.unsqueeze(0).unsqueeze(1)
            
            h_feats, _ = hubert(wav_b)
            sem, pro, spk = encoder(h_feats)
            
            _, _, sem_idx, _ = sem_vq(sem)
            _, _, pro_idx, _ = pro_vq(pro)
            _, _, spk_idx, _ = spk_vq(spk)  # Temporal now
            
            sem_all.append(sem_idx.flatten())
            pro_all.append(pro_idx.flatten())
            spk_all.append(spk_idx.flatten())
        
        sem_all = torch.cat(sem_all, dim=0)
        pro_all = torch.cat(pro_all, dim=0)
        spk_all = torch.cat(spk_all, dim=0)
        
        def branch_stats(indices, name):
            counts = torch.bincount(indices, minlength=args.num_codes)
            num_used = (counts > 0).sum().item()
            probs = counts.float() / counts.sum()
            entropy = -(probs[probs > 0] * torch.log2(probs[probs > 0]) + 1e-10).sum().item()
            print(f"    {name}: {num_used}/{args.num_codes} ({100*num_used/args.num_codes:.1f}%), H={entropy:.2f} bits")
            return num_used, entropy
        
        print(f"  Code stats (train):")
        sem_used, sem_h = branch_stats(sem_all, "Sem")
        pro_used, pro_h = branch_stats(pro_all, "Pro")
        spk_used, spk_h = branch_stats(spk_all, "Spk")

    # Full validation evaluation using prepare.py helpers
    print("  Running full validation...")
    
    fac_for_eval = SimpleFactorizer(encoder).to(device)
    fac_for_eval.eval()
    
    # Use prepare.py's evaluate_vqvae_val
    hp = {"fsq_sem_input_scale": 1.0, "fsq_pro_input_scale": 1.0, "fsq_input_scale": 1.0}
    
    # prepare.py expects VQ to return (z_q, loss, indices) - our VQ returns 4 values
    # Wrap VQ modules to match expected interface
    class VQWrapper(nn.Module):
        def __init__(self, vq):
            super().__init__()
            self.vq = vq
            self.vocab_size = vq.num_codes
        def forward(self, x):
            z_q, loss, indices, _ = self.vq(x)
            return z_q, loss, indices
        def from_indices(self, indices):
            return self.vq.from_indices(indices) if hasattr(self.vq, 'from_indices') else self.vq.embedding(indices)
    
    sem_vq_wrap = VQWrapper(sem_vq)
    pro_vq_wrap = VQWrapper(pro_vq)
    spk_vq_wrap = VQWrapper(spk_vq)
    
    val_metrics = evaluate_vqvae_val(
        hubert, fac_for_eval, decoder, sem_vq_wrap, pro_vq_wrap, spk_vq_wrap,
        val_loader, device, hp, use_amp, stats_batches=16
    )
    
    # Speech metrics
    speech = measure_speech_pesq_stoi(
        val_paths[:12], hubert, fac_for_eval, decoder, sem_vq_wrap, pro_vq_wrap, spk_vq_wrap,
        device, hp, use_amp
    )

    # Save checkpoint
    ckpt_payload = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "sem_vq": sem_vq.state_dict(),
        "pro_vq": pro_vq.state_dict(),
        "spk_vq": spk_vq.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "config": {
            "sem_dim": args.sem_dim,
            "pro_dim": args.pro_dim,
            "spk_dim": args.spk_dim,
            "num_codes": args.num_codes,
        }
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_pt = os.path.join(args.output_dir, "vqvae_branch_last.pt")
    torch.save(ckpt_payload, out_pt)

    t_end = time.time()
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Print summary
    print("---")
    print(f"val_recon_mse:      {float(val_metrics['val_recon_mse']):.6f}")
    print(f"val_score:          {float(val_metrics['val_score']):.6f}")
    
    def fmt(x):
        if isinstance(x, (float, int)) and math.isfinite(x):
            return f"{float(x):.6f}"
        return "nan"
    
    print(f"val_pesq_wb:        {fmt(speech.get('val_pesq_wb', float('nan')))}")
    print(f"val_stoi:           {fmt(speech.get('val_stoi', float('nan')))}")
    print(f"speech_metrics_n:   {int(speech.get('speech_metrics_n', 0))}")
    print(f"sem_h_bits:         {float(val_metrics['sem_h_bits']):.4f}")
    print(f"pro_h_bits:         {float(val_metrics['pro_h_bits']):.4f}")
    print(f"spk_h_bits:         {float(val_metrics['spk_h_bits']):.4f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {optim_step}")
    print(f"checkpoint:         {out_pt}")


if __name__ == "__main__":
    main()
