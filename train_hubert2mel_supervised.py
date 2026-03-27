"""
Train HuBERT2Mel with SUPERVISED mel matching

Uses ground truth mel spectrograms as targets.
This should give much better mel reconstruction than VQ-VAE loss.

Usage: uv run train_hubert2mel_supervised.py --epochs 100
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import amp as torch_amp

from prepare import (
    DEFAULT_DATA_DIR,
    DEFAULT_HUBERT_CKPT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAL_DATA_DIR,
    make_dataloader,
    split_train_val_files,
    verify_assets,
)
from ultra_low_bitrate_codec.models.bithubert import BitHuBERT


class HuBERT2Mel(nn.Module):
    """MLP projection from HuBERT to mel spectrogram."""
    
    def __init__(self, hubert_dim=768, mel_dim=80, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hubert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, mel_dim),
        )
        
    def forward(self, h):
        return self.net(h)


def main():
    parser = argparse.ArgumentParser(description="Train HuBERT2Mel (supervised)")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HuBERT2Mel Supervised Training")
    print(f"  Device: {device}")
    
    # Setup
    data_dir = os.path.abspath(args.data_dir)
    hubert_ckpt = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, hubert_ckpt, hubert_ckpt)

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

    # Load HubERT
    hubert = BitHuBERT(hidden_dim=384, output_dim=768, num_layers=12).to(device).eval()
    hubert.load_state_dict(torch.load(hubert_ckpt, map_location=device))

    # Model
    hubert2mel = HuBERT2Mel(hubert_dim=768, mel_dim=80, hidden_dim=args.hidden_dim).to(device)
    
    optimizer = torch.optim.AdamW(hubert2mel.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mel spectrogram for ground truth
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)

    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        hubert2mel.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for wav in train_loader:
            wav = wav.to(device, non_blocking=True)
            wav_16k = wav.unsqueeze(1)
            
            with torch.no_grad():
                # Ground truth mel
                mel_gt = mel_transform(wav_16k)
                mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))
                if mel_gt.dim() == 4:
                    mel_gt = mel_gt.squeeze(1)
                
                # HuBERT features
                h_feats, _ = hubert(wav_16k)
            
            amp_ctx = torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            
            with amp_ctx:
                # Predict mel
                mel_pred = hubert2mel(h_feats).transpose(1, 2)
                
                # Align lengths
                min_len = min(mel_pred.shape[2], mel_gt.shape[2])
                mel_pred = mel_pred[:, :, :min_len]
                mel_gt = mel_gt[:, :, :min_len]
                
                # MSE + L1 loss
                loss = F.mse_loss(mel_pred, mel_gt) + F.l1_loss(mel_pred, mel_gt)
            
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(hubert2mel.parameters(), 1.0)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        # Validation
        hubert2mel.eval()
        val_loss = 0.0
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
                
                mel_pred = hubert2mel(h_feats).transpose(1, 2)
                
                min_len = min(mel_pred.shape[2], mel_gt.shape[2])
                mel_pred = mel_pred[:, :, :min_len]
                mel_gt = mel_gt[:, :, :min_len]
                
                val_loss += F.mse_loss(mel_pred, mel_gt).item()
                val_batches += 1
        
        val_loss /= max(1, val_batches)
        train_loss = epoch_loss / max(1, num_batches)
        
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "hubert2mel": hubert2mel.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "hubert_dim": 768,
                    "mel_dim": 80,
                    "hidden_dim": args.hidden_dim,
                }
            }
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(args.output_dir, "hubert2mel_supervised_best.pt"))
            print(f"  ✓ Saved best checkpoint (val_mse: {val_loss:.4f})")
    
    print(f"\nTraining complete! Best val MSE: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'hubert2mel_supervised_best.pt')}")


if __name__ == "__main__":
    main()
