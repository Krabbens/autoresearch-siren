"""
SIREN VQ-VAE with BitVocos Vocoder - End-to-End Audio Training

This script trains:
1. VQ-VAE encoder/decoder for HuBERT feature compression
2. BitVocos neural vocoder for audio synthesis

Usage: uv run train_vocos.py
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
import torchaudio
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
    make_dataloader,
    split_train_val_files,
    verify_assets,
)
from ultra_low_bitrate_codec.models.bithubert import BitHuBERT
from ultra_low_bitrate_codec.models.bit_vocos import BitVocos


# ---------------------------------------------------------------------------
# Vector Quantizer (same as before)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Vector Quantizer with commitment loss and codebook reset."""
    
    def __init__(self, num_codes, codebook_dim, beta=0.25, reset_threshold=0.001,
                 num_residuals=1):
        super().__init__()
        self.num_codes = num_codes
        self.codebook_dim = codebook_dim
        self.beta = beta
        self.reset_threshold = reset_threshold
        self.num_residuals = num_residuals
        
        if num_residuals == 1:
            self.embedding = nn.Embedding(num_codes, codebook_dim)
            self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        else:
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_codes, codebook_dim) for _ in range(num_residuals)
            ])
            for emb in self.embeddings:
                emb.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        self.register_buffer("ema_count", torch.ones(num_codes))
        self.register_buffer("ema_weight", torch.ones_like(
            self.embedding.weight if num_residuals == 1 else self.embeddings[0].weight))
        
    def forward(self, z):
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)
        
        if self.num_residuals == 1:
            distances = (
                torch.sum(z_flat ** 2, dim=1, keepdim=True) 
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(z_flat, self.embedding.weight.t())
            )
            indices = torch.argmin(distances, dim=1)
            z_q = self.embedding(indices).reshape(z.shape)
            
            commit_loss = F.mse_loss(z_q.detach(), z) * self.beta
            codebook_loss = F.mse_loss(z_q, z.detach())
            vq_loss = commit_loss + codebook_loss
            
            z_q = z + (z_q - z).detach()
            
            if self.training:
                self._update_usage(indices)
            
            entropy_bits = self._compute_entropy(indices)
            return z_q, vq_loss, indices, entropy_bits
        else:
            residual = z_flat
            z_q_flat = torch.zeros_like(residual)
            total_loss = 0.0
            all_indices = []
            
            for stage in range(self.num_residuals):
                emb = self.embeddings[stage]
                distances = (
                    torch.sum(residual ** 2, dim=1, keepdim=True) 
                    + torch.sum(emb.weight ** 2, dim=1)
                    - 2 * torch.matmul(residual, emb.weight.t())
                )
                indices = torch.argmin(distances, dim=1)
                z_q_stage = emb(indices)
                
                z_q_flat = z_q_flat + z_q_stage
                residual = residual - z_q_stage.detach()
                
                commit_loss = F.mse_loss(z_q_stage.detach(), z_flat) * self.beta
                codebook_loss = F.mse_loss(z_q_stage, z_flat.detach())
                total_loss += (commit_loss + codebook_loss) / self.num_residuals
                
                all_indices.append(indices)
            
            z_q = z_q_flat.reshape(z.shape)
            z_q = z + (z_q_flat.reshape(z.shape) - z).detach()
            
            if self.training:
                self._update_usage(all_indices[0])
            
            entropy_bits = self._compute_entropy(all_indices[0])
            indices = torch.stack(all_indices, dim=-1)
            
            return z_q, total_loss, indices, entropy_bits
    
    def _update_usage(self, indices):
        counts = torch.bincount(indices.flatten(), minlength=self.num_codes)
        self.ema_count = 0.99 * self.ema_count + 0.01 * counts.float()
        
    def _compute_entropy(self, indices):
        counts = torch.bincount(indices.flatten(), minlength=self.num_codes).float()
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 0]
        return -(probs * torch.log2(probs)).sum().item()
    
    def reset_dead_codes(self, z):
        if self.num_residuals == 1:
            if self.ema_count.min() < self.reset_threshold:
                dead_mask = self.ema_count < self.reset_threshold
                num_dead = dead_mask.sum().item()
                z_flat = z.reshape(-1, self.codebook_dim).float()
                rand_idx = torch.randint(0, len(z_flat), (num_dead,), device=z.device)
                with torch.no_grad():
                    self.embedding.weight[dead_mask] = z_flat[rand_idx].to(self.embedding.weight.dtype)
                    self.ema_count[dead_mask] = 1.0
        else:
            if self.ema_count.min() < self.reset_threshold:
                dead_mask = self.ema_count < self.reset_threshold
                num_dead = dead_mask.sum().item()
                z_flat = z.reshape(-1, self.codebook_dim).float()
                rand_idx = torch.randint(0, len(z_flat), (num_dead,), device=z.device)
                with torch.no_grad():
                    self.embeddings[0].weight[dead_mask] = z_flat[rand_idx].to(self.embeddings[0].weight.dtype)
                    self.ema_count[dead_mask] = 1.0


# ---------------------------------------------------------------------------
# Encoder with mel projection for vocoder
# ---------------------------------------------------------------------------

class BranchEncoder(nn.Module):
    """Encoder with separate semantic, prosody, speaker branches."""
    
    def __init__(self, input_dim=768, hidden_dim=512, 
                 sem_dim=128, pro_dim=64, spk_dim=256,
                 sem_compression=1, pro_compression=4, spk_compression=1,
                 speaker_mode="temporal"):
        super().__init__()
        self.input_dim = input_dim
        self.sem_dim = sem_dim
        self.pro_dim = pro_dim
        self.spk_dim = spk_dim
        self.sem_compression = sem_compression
        self.pro_compression = pro_compression
        self.spk_compression = spk_compression
        self.speaker_mode = speaker_mode
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.semantic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, sem_dim),
        )
        
        self.prosody = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, pro_dim),
        )
        
        if speaker_mode == "global":
            self.speaker_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, spk_dim),
            )
            self.speaker_attn = nn.Linear(hidden_dim, 1)
        else:
            self.speaker_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, spk_dim),
            )
        
    def _compress(self, x, compression):
        if compression <= 1:
            return x
        B, T, D = x.shape
        T_out = T // compression
        x = x[:, :T_out * compression, :]
        x = x.reshape(B, T_out, compression, D)
        return x.mean(dim=2)
        
    def forward(self, x):
        h = self.shared(x)
        
        sem = self.semantic(h)
        sem = self._compress(sem, self.sem_compression)
        
        pro = self.prosody(h)
        pro = self._compress(pro, self.pro_compression)
        
        if self.speaker_mode == "global":
            h_spk = self.speaker_encoder(x)
            attn = torch.softmax(self.speaker_attn(h_spk), dim=1)
            spk = torch.sum(h_spk * attn, dim=1)
        else:
            spk = self.speaker_encoder(x)
            spk = self._compress(spk, self.spk_compression)
        
        return sem, pro, spk


class BranchDecoder(nn.Module):
    """Decoder that reconstructs HuBERT features from quantized branches."""
    
    def __init__(self, sem_dim=128, pro_dim=64, spk_dim=256, 
                 hidden_dim=512, output_dim=768,
                 sem_compression=1, pro_compression=4, spk_compression=1,
                 speaker_mode="temporal"):
        super().__init__()
        self.sem_compression = sem_compression
        self.pro_compression = pro_compression
        self.spk_compression = spk_compression
        self.speaker_mode = speaker_mode
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
        
    def _upsample(self, x, target_len, compression):
        if compression <= 1 or x.shape[1] >= target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode='nearest')
        return x.transpose(1, 2)
        
    def forward(self, sem_q, pro_q, spk_q, target_len=None):
        if target_len is None:
            B, T, _ = sem_q.shape
            target_len = T * self.sem_compression
        
        B = sem_q.shape[0]
        
        sem_q = self._upsample(sem_q, target_len, self.sem_compression)
        pro_q = self._upsample(pro_q, target_len, self.pro_compression)
        
        if spk_q.dim() == 2:
            spk_expanded = spk_q.unsqueeze(1).expand(-1, target_len, -1)
        elif spk_q.dim() == 4:
            spk_expanded = spk_q.squeeze(1)
            if spk_expanded.shape[1] != target_len:
                spk_expanded = self._upsample(spk_expanded, target_len, self.spk_compression)
        else:
            spk_expanded = self._upsample(spk_q, target_len, self.spk_compression)
        
        min_len = min(sem_q.shape[1], pro_q.shape[1], spk_expanded.shape[1], target_len)
        sem_q = sem_q[:, :min_len, :]
        pro_q = pro_q[:, :min_len, :]
        spk_expanded = spk_expanded[:, :min_len, :]
        
        combined = torch.cat([sem_q, pro_q, spk_expanded], dim=-1)
        return self.net(combined)


# ---------------------------------------------------------------------------
# Mel Projection for BitVocos
# ---------------------------------------------------------------------------

class HuBERT2Mel(nn.Module):
    """Linear projection from HuBERT features to mel spectrogram."""
    
    def __init__(self, hubert_dim=768, mel_dim=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hubert_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, mel_dim),
        )
        
    def forward(self, h_recon):
        """
        Args:
            h_recon: (B, T, 768) reconstructed HuBERT features
        Returns:
            mel: (B, T, 80) mel spectrogram
        """
        return self.net(h_recon)


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


def main():
    parser = argparse.ArgumentParser(description="VQ-VAE + BitVocos training")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    
    # Architecture hyperparameters
    parser.add_argument("--sem_dim", type=int, default=128)
    parser.add_argument("--sem_num_codes", type=int, default=32)
    parser.add_argument("--sem_compression", type=int, default=2)
    parser.add_argument("--sem_num_residuals", type=int, default=3)
    
    parser.add_argument("--pro_dim", type=int, default=64)
    parser.add_argument("--pro_num_codes", type=int, default=64)
    parser.add_argument("--pro_compression", type=int, default=8)
    parser.add_argument("--pro_num_residuals", type=int, default=1)
    
    parser.add_argument("--spk_dim", type=int, default=256)
    parser.add_argument("--spk_num_codes", type=int, default=32)
    parser.add_argument("--spk_compression", type=int, default=2)
    parser.add_argument("--spk_num_residuals", type=int, default=2)
    
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--speaker_mode", type=str, default="temporal", choices=["temporal", "global"])
    
    # VQ hyperparameters
    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--commitment_beta", type=float, default=1.0)
    parser.add_argument("--reset_threshold", type=float, default=0.0005)
    
    # Vocoder hyperparameters
    parser.add_argument("--vocos_dim", type=int, default=384)
    parser.add_argument("--vocos_layers", type=int, default=6)
    parser.add_argument("--vocoder_weight", type=float, default=0.1)
    
    args = parser.parse_args()

    # Setup paths
    data_dir = os.path.abspath(args.data_dir)
    config_path = os.path.abspath(args.config)
    hubert_ckpt = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, config_path, hubert_ckpt)

    val_dir = args.val_data_dir.strip() if args.val_data_dir else ""
    train_paths, val_paths = split_train_val_files(data_dir, val_data_dir=val_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    base_lr = args.lr
    weight_decay = 0.01
    grad_clip = 0.5
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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
        sem_compression=args.sem_compression,
        pro_compression=args.pro_compression,
        spk_compression=args.spk_compression,
        speaker_mode=args.speaker_mode
    ).to(device)
    
    decoder = BranchDecoder(
        sem_dim=args.sem_dim, pro_dim=args.pro_dim, spk_dim=args.spk_dim,
        hidden_dim=args.hidden_dim, output_dim=768,
        sem_compression=args.sem_compression,
        pro_compression=args.pro_compression,
        spk_compression=args.spk_compression,
        speaker_mode=args.speaker_mode
    ).to(device)
    
    # VQ layers
    sem_vq = VectorQuantizer(num_codes=args.sem_num_codes, codebook_dim=args.sem_dim,
                             beta=args.commitment_beta, reset_threshold=args.reset_threshold,
                             num_residuals=args.sem_num_residuals).to(device)
    pro_vq = VectorQuantizer(num_codes=args.pro_num_codes, codebook_dim=args.pro_dim,
                             beta=args.commitment_beta, reset_threshold=args.reset_threshold,
                             num_residuals=args.pro_num_residuals).to(device)
    spk_vq = VectorQuantizer(num_codes=args.spk_num_codes, codebook_dim=args.spk_dim,
                             beta=args.commitment_beta, reset_threshold=args.reset_threshold,
                             num_residuals=args.spk_num_residuals).to(device)
    
    # Mel projection + BitVocos
    hubert2mel = HuBERT2Mel(hubert_dim=768, mel_dim=80).to(device)
    vocoder = BitVocos(
        input_dim=80, dim=args.vocos_dim, num_layers=args.vocos_layers,
        n_fft=1024, hop_length=320
    ).to(device)

    # Optimizer
    params = (list(encoder.parameters()) + list(decoder.parameters()) + 
              list(sem_vq.parameters()) + list(pro_vq.parameters()) + list(spk_vq.parameters()) +
              list(hubert2mel.parameters()) + list(vocoder.parameters()))
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    
    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    # Mel spectrogram for ground truth
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)

    global_step = 0
    optim_step = 0
    
    # Calculate bitrate
    HUBERT_SR = 50
    sem_bits = math.log2(args.sem_num_codes) * args.sem_num_residuals
    pro_bits = math.log2(args.pro_num_codes) * args.pro_num_residuals
    spk_bits = math.log2(args.spk_num_codes) * args.spk_num_residuals
    
    sem_bps = sem_bits * (HUBERT_SR / args.sem_compression)
    pro_bps = pro_bits * (HUBERT_SR / args.pro_compression)
    spk_bps = spk_bits * (HUBERT_SR / args.spk_compression)
    total_bps = sem_bps + pro_bps + spk_bps
    
    print("VQ-VAE + BitVocos Training")
    print(f"  sem: {args.sem_num_codes} codes × {args.sem_num_residuals} stages, {args.sem_compression}x → {sem_bps:.0f} bps")
    print(f"  pro: {args.pro_num_codes} codes × {args.pro_num_residuals} stages, {args.pro_compression}x → {pro_bps:.0f} bps")
    print(f"  spk: {args.spk_num_codes} codes × {args.spk_num_residuals} stages, {args.spk_compression}x → {spk_bps:.0f} bps")
    print(f"  TOTAL BITRATE: {total_bps:.0f} bps")
    print(f"  BitVocos: dim={args.vocos_dim}, layers={args.vocos_layers}")
    print(f"  TIME_BUDGET: {TIME_BUDGET}s")
    print(f"  train wavs: {len(train_paths)}, val wavs: {len(val_paths)}")

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
        hubert2mel.train()
        vocoder.train()

        wav = _next_wav()
        wav = wav.to(device, non_blocking=True)
        
        # Ground truth mel spectrogram
        # wav: (B, T_audio)
        wav_16k = wav.unsqueeze(1)  # (B, 1, T_audio)
        mel_gt = mel_transform(wav_16k)  # (B, 80, T_mel)
        mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))  # Log mel
        # mel_gt shape: (B, 80, T_mel)
        
        # Debug: print shapes on first step
        if global_step == 0:
            print(f"  Debug shapes: wav={wav.shape}, mel_gt={mel_gt.shape}")
        
        # Squeeze any extra dimensions (dataloader may add batch dim)
        if mel_gt.dim() == 4:
            mel_gt = mel_gt.squeeze(1)  # (B, 80, T_mel)
        
        with torch.no_grad():
            h_feats, _ = hubert(wav_16k)

        amp_ctx = (
            torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            if device.type == "cuda"
            else nullcontext()
        )
        
        with amp_ctx:
            # VQ-VAE
            sem, pro, spk = encoder(h_feats)
            
            sem_q, sem_vq_loss, sem_idx, sem_entropy = sem_vq(sem)
            pro_q, pro_vq_loss, pro_idx, pro_entropy = pro_vq(pro)
            spk_q, spk_vq_loss, spk_idx, spk_entropy = spk_vq(spk)
            
            h_recon = decoder(sem_q, pro_q, spk_q, target_len=h_feats.shape[1])
            
            # HuBERT reconstruction loss
            recon_loss = F.mse_loss(h_recon, h_feats)
            
            # Mel prediction loss
            mel_pred = hubert2mel(h_recon)  # (B, T_h, 80)
            mel_pred = mel_pred.transpose(1, 2)  # (B, 80, T_h)
            
            # mel_gt is (B, 80, T_mel) from audio mel transform
            # T_h and T_mel may differ due to HuBERT frame rate vs mel frame rate
            # Align by taking minimum length
            min_len = min(mel_pred.shape[2], mel_gt.shape[2])
            mel_pred = mel_pred[:, :, :min_len]
            mel_gt = mel_gt[:, :, :min_len]
            mel_loss = F.mse_loss(mel_pred, mel_gt)
            
            # Vocoder loss (audio reconstruction)
            # vocoder expects (B, T, 80) or (B, 80, T)
            mel_pred_transposed = mel_pred.transpose(1, 2)  # (B, 80, T)
            audio_pred = vocoder(mel_pred_transposed)
            
            # Align audio lengths
            target_audio_len = wav.shape[1]
            if audio_pred.shape[1] > target_audio_len:
                audio_pred = audio_pred[:, :target_audio_len]
            elif audio_pred.shape[1] < target_audio_len:
                # Pad if needed
                pad_len = target_audio_len - audio_pred.shape[1]
                audio_pred = F.pad(audio_pred, (0, pad_len))
            
            vocoder_loss = F.mse_loss(audio_pred, wav)
            
            # Total VQ loss
            total_vq_loss = sem_vq_loss + pro_vq_loss + spk_vq_loss
            
            # Total loss
            raw_loss = (
                recon_loss + 
                args.vq_weight * total_vq_loss + 
                args.vocoder_weight * vocoder_loss +
                0.1 * mel_loss
            )

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
            spk_vq.reset_dead_codes(spk)
        
        global_step += 1
        optim_step += 1
        optimizer.zero_grad(set_to_none=True)

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss_f
        deb = smooth_loss / (1.0 - ema_beta ** min(optim_step, 10**9))
        rem = max(0.0, TIME_BUDGET - total_training_time)
        
        avg_entropy = (sem_entropy + pro_entropy + spk_entropy) / 3
        
        print(
            f"\rstep {optim_step:05d} ({100 * progress:.1f}%) | "
            f"loss: {deb:.4f} | recon: {recon_loss.item():.4f} | "
            f"vq: {total_vq_loss.item():.4f} | voc: {vocoder_loss.item():.4f} | "
            f"H: {avg_entropy:.2f} | dt: {dt * 1000:.0f}ms | rem: {rem:.0f}s ",
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
    sem_vq.eval()
    pro_vq.eval()
    spk_vq.eval()
    encoder.eval()
    
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
            _, _, spk_idx, _ = spk_vq(spk)
            
            sem_all.append(sem_idx.flatten())
            pro_all.append(pro_idx.flatten())
            spk_all.append(spk_idx.flatten())
        
        sem_all = torch.cat(sem_all, dim=0)
        pro_all = torch.cat(pro_all, dim=0)
        spk_all = torch.cat(spk_all, dim=0)
        
        def branch_stats(indices, name, num_codes):
            counts = torch.bincount(indices, minlength=num_codes)
            num_used = (counts > 0).sum().item()
            probs = counts.float() / counts.sum()
            entropy = -(probs[probs > 0] * torch.log2(probs[probs > 0]) + 1e-10).sum().item()
            print(f"    {name}: {num_used}/{num_codes} ({100*num_used/num_codes:.1f}%), H={entropy:.2f} bits")
            return num_used, entropy
        
        print(f"  Code stats (train):")
        sem_used, sem_h = branch_stats(sem_all, "Sem", args.sem_num_codes)
        pro_used, pro_h = branch_stats(pro_all, "Pro", args.pro_num_codes)
        spk_used, spk_h = branch_stats(spk_all, "Spk", args.spk_num_codes)

    # Save checkpoint
    ckpt_payload = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "sem_vq": sem_vq.state_dict(),
        "pro_vq": pro_vq.state_dict(),
        "spk_vq": spk_vq.state_dict(),
        "hubert2mel": hubert2mel.state_dict(),
        "vocoder": vocoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "config": {
            "sem_dim": args.sem_dim,
            "sem_num_codes": args.sem_num_codes,
            "sem_num_residuals": args.sem_num_residuals,
            "sem_compression": args.sem_compression,
            "pro_dim": args.pro_dim,
            "pro_num_codes": args.pro_num_codes,
            "pro_num_residuals": args.pro_num_residuals,
            "pro_compression": args.pro_compression,
            "spk_dim": args.spk_dim,
            "spk_num_codes": args.spk_num_codes,
            "spk_num_residuals": args.spk_num_residuals,
            "spk_compression": args.spk_compression,
            "speaker_mode": args.speaker_mode,
            "vocos_dim": args.vocos_dim,
            "vocos_layers": args.vocos_layers,
        }
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_pt = os.path.join(args.output_dir, "vqvae_vocos_last.pt")
    torch.save(ckpt_payload, out_pt)

    t_end = time.time()
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Print summary
    print("---")
    print(f"val_mel_loss:       (eval separately)")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {optim_step}")
    print(f"checkpoint:         {out_pt}")
    print(f"  To evaluate: load checkpoint and run inference with vocoder")


if __name__ == "__main__":
    main()
