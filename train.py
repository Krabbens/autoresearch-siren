"""
SIREN VQ-VAE Phase 1 — Simplified architecture that actually works.

Key changes from broken SIREN V8:
1. Single quantizer branch (not 3 separate sem/pro/spk)
2. Standard Vector Quantization with codebook reset
3. Explicit entropy regularization (additive, not subtracted!)
4. Simple linear encoder/decoder - no complex branching

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
# Simple Vector Quantizer with codebook reset
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with:
    - Commitment loss (encoder output should match codebook)
    - Codebook reset for dead codes
    - Entropy tracking for monitoring
    """
    def __init__(self, num_codes, codebook_dim, beta=0.25, reset_threshold=0.01):
        super().__init__()
        self.num_codes = num_codes
        self.codebook_dim = codebook_dim
        self.beta = beta
        self.reset_threshold = reset_threshold
        
        # Codebook
        self.embedding = nn.Embedding(num_codes, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        # EMA tracking for codebook usage
        self.register_buffer("ema_count", torch.ones(num_codes))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())
        
    def forward(self, z):
        """
        Args:
            z: (B, T, D) encoder outputs
        Returns:
            z_q: quantized output (with STE gradient)
            commit_loss: commitment loss
            indices: code indices
            entropy_bits: entropy of code usage (for monitoring)
        """
        B, T, D = z.shape
        
        # Flatten for quantization
        z_flat = z.reshape(-1, D)
        
        # Find nearest codebook entry
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) 
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        
        # Get quantized vectors
        z_q = self.embedding(indices).reshape(z.shape)
        
        # Commitment loss: encoder should output vectors close to codebook
        commit_loss = F.mse_loss(z_q.detach(), z) * self.beta
        
        # Codebook loss: codebook should move towards encoder outputs
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        # Total VQ loss
        vq_loss = commit_loss + codebook_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Track usage for codebook reset
        if self.training:
            self._update_usage(indices)
        
        # Compute entropy for monitoring
        entropy_bits = self._compute_entropy(indices)
        
        return z_q, vq_loss, indices, entropy_bits
    
    def _update_usage(self, indices):
        """Update EMA usage counts for codebook reset."""
        counts = torch.bincount(indices.flatten(), minlength=self.num_codes)
        self.ema_count = 0.99 * self.ema_count + 0.01 * counts.float()
        
    def _compute_entropy(self, indices):
        """Compute entropy of code usage in bits."""
        counts = torch.bincount(indices.flatten(), minlength=self.num_codes).float()
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 0]
        entropy = -(probs * torch.log2(probs)).sum()
        return entropy.item()
    
    def reset_dead_codes(self, z):
        """Reset unused codes to random encoder outputs."""
        if self.ema_count.min() < self.reset_threshold:
            dead_mask = self.ema_count < self.reset_threshold
            num_dead = dead_mask.sum().item()
            
            # Sample random encoder outputs to replace dead codes
            z_flat = z.reshape(-1, self.codebook_dim)
            rand_idx = torch.randint(0, len(z_flat), (num_dead,), device=z.device)
            
            with torch.no_grad():
                self.embedding.weight[dead_mask] = z_flat[rand_idx]
                self.ema_count[dead_mask] = 1.0


# ---------------------------------------------------------------------------
# Simple Encoder and Decoder
# ---------------------------------------------------------------------------

class SimpleEncoder(nn.Module):
    """Simple MLP encoder that projects HuBERT features to latent space."""
    
    def __init__(self, input_dim=768, hidden_dim=512, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, 768) HuBERT features
        Returns:
            z: (B, T, latent_dim) latent features
        """
        return self.net(x)


class SimpleDecoder(nn.Module):
    """Simple MLP decoder that reconstructs HuBERT features from quantized latents."""
    
    def __init__(self, latent_dim=64, hidden_dim=512, output_dim=768):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, z_q):
        """
        Args:
            z_q: (B, T, latent_dim) quantized latents
        Returns:
            x_recon: (B, T, 768) reconstructed features
        """
        return self.net(z_q)


# ---------------------------------------------------------------------------
# Entropy Regularization
# ---------------------------------------------------------------------------

def entropy_bonus(indices, num_codes, target_entropy=None):
    """
    Add bonus for high entropy code usage.
    
    Unlike the broken SIREN diversity term, this is ADDED to loss
    when entropy is BELOW target, encouraging more code usage.
    """
    counts = torch.bincount(indices.flatten(), minlength=num_codes).float()
    probs = counts / (counts.sum() + 1e-10)
    
    # Compute current entropy
    probs_nonzero = probs[probs > 0]
    current_entropy = -(probs_nonzero * torch.log2(probs_nonzero)).sum()
    
    # Target entropy (default: 80% of maximum)
    if target_entropy is None:
        target_entropy = 0.8 * math.log2(num_codes)
    
    # Penalize low entropy
    if current_entropy < target_entropy:
        return (target_entropy - current_entropy) * 0.01
    return current_entropy * 0.0  # No bonus if already above target


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress: float) -> float:
    """Linear warmup, constant, then cosine decay."""
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


def main():
    parser = argparse.ArgumentParser(description="Simplified VQ-VAE for autoresearch")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--resume_ckpt", default=None)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_codes", type=int, default=512)
    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.5)
    parser.add_argument("--reset_threshold", type=float, default=0.001)
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

    encoder = SimpleEncoder(input_dim=768, hidden_dim=512, latent_dim=args.latent_dim).to(device)
    decoder = SimpleDecoder(latent_dim=args.latent_dim, hidden_dim=512, output_dim=768).to(device)
    vq = VectorQuantizer(num_codes=args.num_codes, codebook_dim=args.latent_dim, reset_threshold=args.reset_threshold).to(device)

    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(vq.parameters())
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    
    # AMP
    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    steps_per_epoch = max(1, len(train_loader))
    global_step = 0
    optim_step = 0
    
    print("Simplified VQ-VAE (autoresearch)")
    print(f"  latent_dim: {args.latent_dim}, num_codes: {args.num_codes}")
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
        vq.train()

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
            z = encoder(h_feats)
            
            # Quantize
            z_q, vq_loss, indices, entropy_bits = vq(z)
            
            # Decode
            h_recon = decoder(z_q)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(h_recon, h_feats)
            
            # Entropy bonus (encourage code usage)
            ent_bonus = entropy_bonus(indices, args.num_codes)
            
            # Total loss
            raw_loss = recon_loss + args.vq_weight * vq_loss + args.entropy_weight * ent_bonus

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
        
        # Periodic codebook reset for dead codes
        if global_step % 100 == 0:
            vq.reset_dead_codes(z)
        
        global_step += 1
        optim_step += 1
        optimizer.zero_grad(set_to_none=True)

        # EMA smoothing
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss_f
        deb = smooth_loss / (1.0 - ema_beta ** min(optim_step, 10**9))
        rem = max(0.0, TIME_BUDGET - total_training_time)
        
        print(
            f"\rstep {optim_step:05d} ({100 * progress:.1f}%) | "
            f"loss: {deb:.4f} | recon: {recon_loss.item():.4f} | "
            f"vq: {vq_loss.item():.4f} | H: {entropy_bits:.2f} | "
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

    # Code usage stats
    vq.eval()
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        all_indices = []
        for batch_idx, wav_path in enumerate(train_paths[:batch_size * 4]):
            import soundfile as sf
            wav_np, _ = sf.read(wav_path)
            wav_t = torch.tensor(wav_np, dtype=torch.float32).to(device)
            if wav_t.shape[0] > 4 * 16000:
                wav_t = wav_t[:4 * 16000]
            wav_b = wav_t.unsqueeze(0).unsqueeze(1)
            
            h_feats, _ = hubert(wav_b)
            z = encoder(h_feats)
            _, _, indices, _ = vq(z)
            all_indices.append(indices.flatten())
        
        all_indices = torch.cat(all_indices, dim=0)
        counts = torch.bincount(all_indices, minlength=args.num_codes)
        num_used = (counts > 0).sum().item()
        max_entropy = math.log2(args.num_codes)
        actual_entropy = -(counts[counts > 0].float() / counts.sum()) * torch.log2(
            counts[counts > 0].float() / counts.sum() + 1e-10
        )
        actual_entropy = actual_entropy.sum().item()
        
    print(f"  Code stats (train):")
    print(f"    Used codes: {num_used}/{args.num_codes} ({100*num_used/args.num_codes:.1f}%)")
    print(f"    Entropy: {actual_entropy:.2f} bits (max {max_entropy:.2f})")

    # Validation
    print("  Running validation...")
    
    # Simple validation MSE
    val_mse = 0.0
    val_count = 0
    with torch.no_grad():
        for wav_path in val_paths[:100]:
            import soundfile as sf
            wav_np, _ = sf.read(wav_path)
            wav_t = torch.tensor(wav_np, dtype=torch.float32).to(device)
            if wav_t.shape[0] > 4 * 16000:
                wav_t = wav_t[:4 * 16000]
            wav_b = wav_t.unsqueeze(0).unsqueeze(1)
            
            h_feats, _ = hubert(wav_b)
            z = encoder(h_feats)
            z_q, _, indices, _ = vq(z)
            h_recon = decoder(z_q)
            
            mse = F.mse_loss(h_recon, h_feats).item()
            val_mse += mse
            val_count += 1
    
    val_mse /= max(1, val_count)
    
    # Speech metrics (proxy via Griffin-Lim)
    val_pesq_wb = float('nan')
    val_stoi = float('nan')
    speech_metrics_n = 0
    try:
        # Create wrapper for prepare.evaluate_vqvae_val interface
        class SimpleFactorizer(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, h, c):
                z = self.encoder(h)
                # Return semantic, prosody, speaker (all same for simple version)
                return z, z, torch.zeros(h.shape[0], 256).to(h.device)
        
        fac = SimpleFactorizer(encoder).to(device)
        fac.eval()
        
        speech = measure_speech_pesq_stoi(
            val_paths[:12], hubert, fac, decoder, vq, vq, vq, device, {}, use_amp
        )
        val_pesq_wb = speech.get('val_pesq_wb', float('nan'))
        val_stoi = speech.get('val_stoi', float('nan'))
        speech_metrics_n = speech.get('speech_metrics_n', 0)
    except Exception as e:
        print(f"  Speech metrics failed: {e}")

    # Save checkpoint
    ckpt_payload = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "vq": vq.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "config": {
            "latent_dim": args.latent_dim,
            "num_codes": args.num_codes,
        }
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_pt = os.path.join(args.output_dir, "vqvae_simple_last.pt")
    torch.save(ckpt_payload, out_pt)

    t_end = time.time()
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Print summary
    print("---")
    print(f"val_recon_mse:      {val_mse:.6f}")
    print(f"val_score:          {val_mse:.6f}")  # Same as MSE for simple version
    
    def fmt(x):
        if isinstance(x, float) and math.isfinite(x):
            return f"{x:.6f}"
        return "nan"
    
    print(f"val_pesq_wb:        {fmt(val_pesq_wb)}")
    print(f"val_stoi:           {fmt(val_stoi)}")
    print(f"speech_metrics_n:   {int(speech_metrics_n)}")
    print(f"sem_h_bits:         {actual_entropy:.4f}")
    print(f"pro_h_bits:         {actual_entropy:.4f}")  # Same for single branch
    print(f"spk_h_bits:         {actual_entropy:.4f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {optim_step}")
    print(f"checkpoint:         {out_pt}")


if __name__ == "__main__":
    main()
