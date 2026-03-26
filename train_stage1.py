"""
Stage 1: Train VQ-VAE + HuBERT2Mel projection

This stage focuses on learning good mel spectrogram reconstruction.
The vocoder is NOT trained yet - we use ground truth mel for evaluation.

Usage: uv run train_stage1.py --epochs 10
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from torch import amp as torch_amp

from prepare import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_HUBERT_CKPT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAL_DATA_DIR,
    make_dataloader,
    split_train_val_files,
    verify_assets,
)
from ultra_low_bitrate_codec.models.bithubert import BitHuBERT


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, codebook_dim, beta=0.25, num_residuals=1):
        super().__init__()
        self.num_codes = num_codes
        self.codebook_dim = codebook_dim
        self.beta = beta
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
            
            return z_q, vq_loss, indices
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
            indices = torch.stack(all_indices, dim=-1)
            
            if self.training:
                self._update_usage(all_indices[0])
            
            return z_q, total_loss, indices
    
    def _update_usage(self, indices):
        counts = torch.bincount(indices.flatten(), minlength=self.num_codes)
        self.ema_count = 0.99 * self.ema_count + 0.01 * counts.float()


class BranchEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, 
                 sem_dim=128, pro_dim=64, spk_dim=256,
                 sem_compression=1, pro_compression=4, spk_compression=1):
        super().__init__()
        self.sem_compression = sem_compression
        self.pro_compression = pro_compression
        self.spk_compression = spk_compression
        
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
        
        spk = self.speaker_encoder(x)
        spk = self._compress(spk, self.spk_compression)
        
        return sem, pro, spk


class BranchDecoder(nn.Module):
    def __init__(self, sem_dim=128, pro_dim=64, spk_dim=256, 
                 hidden_dim=512, output_dim=768,
                 sem_compression=1, pro_compression=4, spk_compression=1):
        super().__init__()
        self.sem_compression = sem_compression
        self.pro_compression = pro_compression
        self.spk_compression = spk_compression
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
        
        sem_q = self._upsample(sem_q, target_len, self.sem_compression)
        pro_q = self._upsample(pro_q, target_len, self.pro_compression)
        spk_expanded = self._upsample(spk_q, target_len, self.spk_compression)
        
        min_len = min(sem_q.shape[1], pro_q.shape[1], spk_expanded.shape[1], target_len)
        combined = torch.cat([
            sem_q[:, :min_len, :],
            pro_q[:, :min_len, :],
            spk_expanded[:, :min_len, :]
        ], dim=-1)
        return self.net(combined)


class HuBERT2Mel(nn.Module):
    """Linear projection from HuBERT features to mel spectrogram."""
    
    def __init__(self, hubert_dim=768, mel_dim=80, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hubert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mel_dim),
        )
        
    def forward(self, h_recon):
        return self.net(h_recon)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: VQ-VAE + Mel Projection")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Architecture
    parser.add_argument("--sem_dim", type=int, default=128)
    parser.add_argument("--sem_num_codes", type=int, default=32)
    parser.add_argument("--sem_num_residuals", type=int, default=3)
    parser.add_argument("--sem_compression", type=int, default=2)
    
    parser.add_argument("--pro_dim", type=int, default=64)
    parser.add_argument("--pro_num_codes", type=int, default=64)
    parser.add_argument("--pro_num_residuals", type=int, default=1)
    parser.add_argument("--pro_compression", type=int, default=8)
    
    parser.add_argument("--spk_dim", type=int, default=256)
    parser.add_argument("--spk_num_codes", type=int, default=32)
    parser.add_argument("--spk_num_residuals", type=int, default=2)
    parser.add_argument("--spk_compression", type=int, default=2)
    
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--mel_hidden_dim", type=int, default=512)
    
    # Loss weights
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--mel_weight", type=float, default=10.0)  # Higher weight for mel
    parser.add_argument("--commitment_beta", type=float, default=1.0)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 1 Training: VQ-VAE + Mel Projection")
    print(f"  Device: {device}")
    
    # Setup
    data_dir = os.path.abspath(args.data_dir)
    config_path = os.path.abspath(args.config)
    hubert_ckpt = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, config_path, hubert_ckpt)

    val_dir = args.val_data_dir.strip() if args.val_data_dir else ""
    train_paths, val_paths = split_train_val_files(data_dir, val_data_dir=val_dir)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pin = device.type == "cuda"
    train_loader = make_dataloader(
        train_paths, args.batch_size, shuffle=True, num_workers=4,
        prefetch_factor=2, pin_memory=pin, seed=int(time.time()) % (2**31)
    )
    val_loader = make_dataloader(
        val_paths, args.batch_size, shuffle=False, num_workers=2,
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
    ).to(device)
    
    decoder = BranchDecoder(
        sem_dim=args.sem_dim, pro_dim=args.pro_dim, spk_dim=args.spk_dim,
        hidden_dim=args.hidden_dim, output_dim=768,
        sem_compression=args.sem_compression,
        pro_compression=args.pro_compression,
        spk_compression=args.spk_compression,
    ).to(device)
    
    sem_vq = VectorQuantizer(
        num_codes=args.sem_num_codes, codebook_dim=args.sem_dim,
        beta=args.commitment_beta, num_residuals=args.sem_num_residuals
    ).to(device)
    
    pro_vq = VectorQuantizer(
        num_codes=args.pro_num_codes, codebook_dim=args.pro_dim,
        beta=args.commitment_beta, num_residuals=args.pro_num_residuals
    ).to(device)
    
    spk_vq = VectorQuantizer(
        num_codes=args.spk_num_codes, codebook_dim=args.spk_dim,
        beta=args.commitment_beta, num_residuals=args.spk_num_residuals
    ).to(device)
    
    hubert2mel = HuBERT2Mel(
        hubert_dim=768, mel_dim=80, hidden_dim=args.mel_hidden_dim
    ).to(device)

    # Optimizer
    params = (list(encoder.parameters()) + list(decoder.parameters()) + 
              list(sem_vq.parameters()) + list(pro_vq.parameters()) + 
              list(spk_vq.parameters()) + list(hubert2mel.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    
    # Mel spectrogram for ground truth
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)

    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    best_val_mel_loss = float('inf')
    
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        sem_vq.train()
        pro_vq.train()
        spk_vq.train()
        hubert2mel.train()
        
        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_recon_loss = 0.0
        num_batches = 0
        
        for wav in train_loader:
            wav = wav.to(device, non_blocking=True)
            
            # Ground truth mel
            wav_16k = wav.unsqueeze(1)
            mel_gt = mel_transform(wav_16k)  # (B, 80, T_mel)
            mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))
            
            # Squeeze if dataloader adds extra dim
            if mel_gt.dim() == 4:
                mel_gt = mel_gt.squeeze(1)
            
            with torch.no_grad():
                h_feats, _ = hubert(wav_16k)

            amp_ctx = torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if device.type == "cuda" else nullcontext()
            
            with amp_ctx:
                # VQ-VAE
                sem, pro, spk = encoder(h_feats)
                
                sem_q, sem_vq_loss, _ = sem_vq(sem)
                pro_q, pro_vq_loss, _ = pro_vq(pro)
                spk_q, spk_vq_loss, _ = spk_vq(spk)
                
                h_recon = decoder(sem_q, pro_q, spk_q, target_len=h_feats.shape[1])
                
                # Mel prediction
                mel_pred = hubert2mel(h_recon)  # (B, T_h, 80)
                mel_pred = mel_pred.transpose(1, 2)  # (B, 80, T_h)
                
                # Align lengths
                min_len = min(mel_pred.shape[2], mel_gt.shape[2])
                mel_pred = mel_pred[:, :, :min_len]
                mel_gt = mel_gt[:, :, :min_len]
                
                # Losses
                recon_loss = F.mse_loss(h_recon, h_feats)
                mel_loss = F.mse_loss(mel_pred, mel_gt)
                total_vq_loss = sem_vq_loss + pro_vq_loss + spk_vq_loss
                
                loss = (
                    args.recon_weight * recon_loss +
                    args.vq_weight * total_vq_loss +
                    args.mel_weight * mel_loss
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            epoch_mel_loss += mel_loss.item()
            epoch_recon_loss += recon_loss.item()
            num_batches += 1
        
        # Validation
        encoder.eval()
        decoder.eval()
        sem_vq.eval()
        pro_vq.eval()
        spk_vq.eval()
        hubert2mel.eval()
        
        val_mel_loss = 0.0
        val_recon_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for wav in val_loader:
                wav = wav.to(device, non_blocking=True)
                wav_16k = wav.unsqueeze(1)
                mel_gt = mel_transform(wav_16k)
                mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))
                if mel_gt.dim() == 4:
                    mel_gt = mel_gt.squeeze(1)
                
                h_feats, _ = hubert(wav_16k)
                
                sem, pro, spk = encoder(h_feats)
                sem_q, _, _ = sem_vq(sem)
                pro_q, _, _ = pro_vq(pro)
                spk_q, _, _ = spk_vq(spk)
                
                h_recon = decoder(sem_q, pro_q, spk_q, target_len=h_feats.shape[1])
                
                mel_pred = hubert2mel(h_recon).transpose(1, 2)
                
                min_len = min(mel_pred.shape[2], mel_gt.shape[2])
                mel_pred = mel_pred[:, :, :min_len]
                mel_gt = mel_gt[:, :, :min_len]
                
                val_mel_loss += F.mse_loss(mel_pred, mel_gt).item()
                val_recon_loss += F.mse_loss(h_recon, h_feats).item()
                val_batches += 1
        
        val_mel_loss /= max(1, val_batches)
        val_recon_loss /= max(1, val_batches)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train: loss={epoch_loss/num_batches:.4f}, mel={epoch_mel_loss/num_batches:.4f}, recon={epoch_recon_loss/num_batches:.4f}")
        print(f"  Val:   mel={val_mel_loss:.4f}, recon={val_recon_loss:.4f}")
        
        if val_mel_loss < best_val_mel_loss:
            best_val_mel_loss = val_mel_loss
            # Save checkpoint
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "sem_vq": sem_vq.state_dict(),
                "pro_vq": pro_vq.state_dict(),
                "spk_vq": spk_vq.state_dict(),
                "hubert2mel": hubert2mel.state_dict(),
                "optimizer": optimizer.state_dict(),
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
                    "hidden_dim": args.hidden_dim,
                    "mel_hidden_dim": args.mel_hidden_dim,
                },
                "val_mel_loss": val_mel_loss,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(args.output_dir, "stage1_best.pt"))
            print(f"  ✓ Saved best checkpoint (mel loss: {val_mel_loss:.4f})")
        
        print()
    
    print(f"\nStage 1 complete! Best val mel loss: {best_val_mel_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'stage1_best.pt')}")
    print("\nNext: Run train_stage2.py to train the vocoder")


if __name__ == "__main__":
    from contextlib import nullcontext
    main()
