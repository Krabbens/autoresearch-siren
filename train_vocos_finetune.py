"""
Train HuBERT2Mel + BitVocos using SIREN's pre-trained VQ-VAE

This freezes the VQ-VAE and only trains the projection + vocoder.
Much faster convergence than training everything from scratch.

Usage: uv run train_vocos_finetune.py --epochs 50
"""

from __future__ import annotations

import argparse
import math
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
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import FeatureReconstructorV2
from ultra_low_bitrate_codec.utils.v8_residual_fsq import build_v8_residual_fsqs
from ultra_low_bitrate_codec.models.bit_vocos import BitVocos


class HuBERT2Mel(nn.Module):
    """MLP projection from HuBERT to mel spectrogram."""
    
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
# Discriminators
# ---------------------------------------------------------------------------

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in [2, 3, 5, 7, 11]
        ])
        
    def forward(self, x):
        scores = []
        for d in self.discriminators:
            score = d(x)
            scores.append(score)
        return scores


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        B, T = x.shape
        if T % self.period != 0:
            x = F.pad(x, (0, self.period - T % self.period))
        x = x.view(B, 1, -1, self.period)
        
        for conv in self.convs:
            x = self.act(conv(x))
        
        x = self.conv_post(x)
        x = x.flatten(1, -1)
        
        return x


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ResDiscriminator(n_fft) for n_fft in [2048, 1024, 512]
        ])
        
    def forward(self, x):
        scores = []
        for d in self.discriminators:
            score = d(x)
            scores.append(score)
        return scores


class ResDiscriminator(nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.win_length = n_fft
        self.register_buffer('window', torch.hann_window(n_fft))
        
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 32, (3, 9), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
        ])
        self.conv_post = nn.Conv2d(32, 1, (3, 3), padding=(1, 1))
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            self.window, return_complex=True
        )
        spec = torch.view_as_real(spec)
        spec = spec.permute(0, 3, 1, 2)
        
        for conv in self.convs:
            spec = self.act(conv(spec))
        
        spec = self.conv_post(spec)
        spec = spec.flatten(1, -1)
        
        return spec


def discriminator_loss(scores_real, scores_fake):
    loss_real = 0.0
    loss_fake = 0.0
    for real, fake in zip(scores_real, scores_fake):
        loss_real += torch.mean(F.relu(1.0 - real))
        loss_fake += torch.mean(F.relu(1.0 + fake))
    return loss_real + loss_fake


def generator_loss(scores_fake):
    loss = 0.0
    for fake in scores_fake:
        loss += torch.mean(F.relu(1.0 - fake))
    return loss


def feature_matching_loss(features_real, features_fake, weight=10.0):
    loss = 0.0
    for real, fake in zip(features_real, features_fake):
        loss += F.l1_loss(fake, real.detach()) * weight
    return loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train HuBERT2Mel + BitVocos (frozen VQ-VAE)")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--vqvae_ckpt", type=str, default="/home/sperm/siren/SIREN/checkpoints/bit_diffusion_v8_vqvae/vqvae_ep80.pt")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocos_dim", type=int, default=512)
    parser.add_argument("--vocos_layers", type=int, default=8)
    parser.add_argument("--gan_start_epoch", type=int, default=5)
    parser.add_argument("--mel_weight", type=float, default=45.0)
    parser.add_argument("--audio_weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HuBERT2Mel + BitVocos Training (Frozen VQ-VAE)")
    print(f"  Device: {device}")
    print(f"  VQ-VAE: {args.vqvae_ckpt}")
    
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

    # Load SIREN's pre-trained VQ-VAE (frozen)
    config_path = "/home/sperm/siren/SIREN/src/ultra_low_bitrate_codec/configs/ultra58bps_16k.yaml"
    with open(config_path) as f:
        import yaml
        config = yaml.safe_load(f)
    
    fac = InformationFactorizerV2(config).to(device).eval()
    rec = FeatureReconstructorV2(config).to(device).eval()
    sem_vq, pro_vq, spk_vq = build_v8_residual_fsqs(config, device)
    
    vqvae_ckpt = torch.load(args.vqvae_ckpt, map_location=device)
    fac.load_state_dict(vqvae_ckpt["factorizer"])
    rec.load_state_dict(vqvae_ckpt["reconstructor"])
    sem_vq.load_state_dict(vqvae_ckpt["sem_vq"])
    pro_vq.load_state_dict(vqvae_ckpt["pro_vq"])
    spk_vq.load_state_dict(vqvae_ckpt["spk_vq"])
    
    print(f"  Loaded SIREN VQ-VAE (epoch 80)")
    
    # Trainable models
    hubert2mel = HuBERT2Mel(hubert_dim=768, mel_dim=80, hidden_dim=512).to(device)
    
    vocoder = BitVocos(
        input_dim=80, dim=args.vocos_dim, num_layers=args.vocos_layers,
        n_fft=1024, hop_length=320
    ).to(device)
    
    # Discriminators
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    
    # Optimizers (only hubert2mel + vocoder + discriminators)
    proj_opt = torch.optim.AdamW(hubert2mel.parameters(), lr=args.lr, betas=(0.8, 0.99))
    vocoder_opt = torch.optim.AdamW(vocoder.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mpd_opt = torch.optim.AdamW(mpd.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mrd_opt = torch.optim.AdamW(mrd.parameters(), lr=args.lr, betas=(0.8, 0.99))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(proj_opt, T_max=args.epochs)
    
    # Mel spectrogram for ground truth
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)

    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        hubert2mel.train()
        vocoder.train()
        
        train_gan = epoch >= args.gan_start_epoch
        
        if train_gan:
            mpd.train()
            mrd.train()
        else:
            mpd.eval()
            mrd.eval()
        
        epoch_mel_loss = 0.0
        epoch_audio_loss = 0.0
        epoch_gan_loss = 0.0
        num_batches = 0
        
        for wav in train_loader:
            wav = wav.to(device, non_blocking=True)
            
            # Ground truth mel
            wav_16k = wav.unsqueeze(1)
            with torch.no_grad():
                mel_gt = mel_transform(wav_16k)
                mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))
                if mel_gt.dim() == 4:
                    mel_gt = mel_gt.squeeze(1)
                
                # VQ-VAE forward (frozen)
                h_feats, cnn_feats = hubert(wav_16k)
                sem, pro, spk = fac(h_feats, cnn_feats)
                
                # Get FSQ scales from config
                fsem = float(config.get("training", {}).get("fsq_sem_input_scale", 2.95))
                fpro = float(config.get("training", {}).get("fsq_pro_input_scale", 4.15))
                
                sem_z, _, _ = sem_vq(sem * fsem)
                pro_z, _, _ = pro_vq(pro * fpro)
                spk_z, _, _ = spk_vq(spk)
                
                h_recon = rec(sem_z, pro_z, spk_z, target_len=h_feats.shape[1])
            
            # Trainable: HuBERT2Mel + Vocoder
            amp_ctx = torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            
            with amp_ctx:
                # Mel prediction
                mel_pred = hubert2mel(h_recon).transpose(1, 2)
                
                # Align lengths
                min_len = min(mel_pred.shape[2], mel_gt.shape[2])
                mel_pred = mel_pred[:, :, :min_len]
                mel_gt = mel_gt[:, :, :min_len]
                
                mel_loss = F.l1_loss(mel_pred, mel_gt)
                
                # Audio synthesis
                mel_pred_exp = torch.exp(torch.clamp(mel_pred, min=1e-5))
                audio_pred = vocoder(mel_pred_exp)
                
                # Align audio lengths
                target_len = wav.shape[1]
                if audio_pred.shape[1] > target_len:
                    audio_pred = audio_pred[:, :target_len]
                elif audio_pred.shape[1] < target_len:
                    audio_pred = F.pad(audio_pred, (0, target_len - audio_pred.shape[1]))
                
                audio_loss = F.l1_loss(audio_pred, wav)
                
                # GAN loss
                gan_loss = torch.tensor(0.0, device=device)
                feat_loss = torch.tensor(0.0, device=device)
                
                if train_gan:
                    # Train discriminators
                    mpd_opt.zero_grad()
                    mrd_opt.zero_grad()
                    
                    scores_real_mpd = mpd(wav)
                    scores_real_mrd = mrd(wav)
                    scores_fake_mpd = mpd(audio_pred.detach())
                    scores_fake_mrd = mrd(audio_pred.detach())
                    
                    loss_d_mpd = discriminator_loss(scores_real_mpd, scores_fake_mpd)
                    loss_d_mrd = discriminator_loss(scores_real_mrd, scores_fake_mrd)
                    
                    if use_amp:
                        scaler.scale(loss_d_mpd).backward()
                        scaler.scale(loss_d_mrd).backward()
                        scaler.step(mpd_opt)
                        scaler.step(mrd_opt)
                        scaler.update()
                    else:
                        loss_d_mpd.backward()
                        loss_d_mrd.backward()
                        mpd_opt.step()
                        mrd_opt.step()
                    
                    # Generator loss
                    scores_fake_mpd = mpd(audio_pred)
                    scores_fake_mrd = mrd(audio_pred)
                    
                    gan_loss = generator_loss(scores_fake_mpd) + generator_loss(scores_fake_mrd)
                    
                    # Feature matching
                    with torch.no_grad():
                        scores_real_mpd = mpd(wav)
                        scores_real_mrd = mrd(wav)
                    
                    feat_loss = (
                        feature_matching_loss(scores_fake_mpd, scores_real_mpd, 10.0) +
                        feature_matching_loss(scores_fake_mrd, scores_real_mrd, 10.0)
                    )
                
                # Total loss
                total_loss = args.mel_weight * mel_loss + args.audio_weight * audio_loss + gan_loss + feat_loss
            
            # Optimize
            proj_opt.zero_grad()
            vocoder_opt.zero_grad()
            
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(proj_opt)
                scaler.unscale_(vocoder_opt)
            else:
                total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(hubert2mel.parameters()) + list(vocoder.parameters()), 0.5)
            
            if use_amp:
                scaler.step(proj_opt)
                scaler.step(vocoder_opt)
                scaler.update()
            else:
                proj_opt.step()
                vocoder_opt.step()
            
            epoch_mel_loss += mel_loss.item()
            epoch_audio_loss += audio_loss.item()
            if train_gan:
                epoch_gan_loss += gan_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        # Validation
        hubert2mel.eval()
        vocoder.eval()
        
        val_mel_loss = 0.0
        val_audio_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for wav in val_loader:
                wav = wav.to(device, non_blocking=True)
                wav_16k = wav.unsqueeze(1)
                
                mel_gt = mel_transform(wav_16k)
                mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))
                if mel_gt.dim() == 4:
                    mel_gt = mel_gt.squeeze(1)
                
                h_feats, cnn_feats = hubert(wav_16k)
                sem, pro, spk = fac(h_feats, cnn_feats)
                
                fsem = float(config.get("training", {}).get("fsq_sem_input_scale", 2.95))
                fpro = float(config.get("training", {}).get("fsq_pro_input_scale", 4.15))
                
                sem_z, _, _ = sem_vq(sem * fsem)
                pro_z, _, _ = pro_vq(pro * fpro)
                spk_z, _, _ = spk_vq(spk)
                
                h_recon = rec(sem_z, pro_z, spk_z, target_len=h_feats.shape[1])
                
                mel_pred = hubert2mel(h_recon).transpose(1, 2)
                
                min_len = min(mel_pred.shape[2], mel_gt.shape[2])
                mel_pred = mel_pred[:, :, :min_len]
                mel_gt = mel_gt[:, :, :min_len]
                
                val_mel_loss += F.l1_loss(mel_pred, mel_gt).item()
                
                mel_pred_exp = torch.exp(torch.clamp(mel_pred, min=1e-5))
                audio_pred = vocoder(mel_pred_exp)
                
                if audio_pred.shape[1] > wav.shape[1]:
                    audio_pred = audio_pred[:, :wav.shape[1]]
                elif audio_pred.shape[1] < wav.shape[1]:
                    audio_pred = F.pad(audio_pred, (0, wav.shape[1] - audio_pred.shape[1]))
                
                val_audio_loss += F.l1_loss(audio_pred, wav).item()
                val_batches += 1
        
        val_mel_loss /= max(1, val_batches)
        val_audio_loss /= max(1, val_batches)
        train_mel_loss = epoch_mel_loss / max(1, num_batches)
        train_audio_loss = epoch_audio_loss / max(1, num_batches)
        
        if train_gan:
            train_gan_loss = epoch_gan_loss / max(1, num_batches)
            print(f"Epoch {epoch+1}/{args.epochs}: mel={train_mel_loss:.4f}, audio={train_audio_loss:.4f}, gan={train_gan_loss:.4f}, val_mel={val_mel_loss:.4f}, val_audio={val_audio_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}: mel={train_mel_loss:.4f}, audio={train_audio_loss:.4f} (warmup), val_mel={val_mel_loss:.4f}, val_audio={val_audio_loss:.4f}")
        
        # Save checkpoint
        val_loss = val_mel_loss + val_audio_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "hubert2mel": hubert2mel.state_dict(),
                "vocoder": vocoder.state_dict(),
                "mpd": mpd.state_dict(),
                "mrd": mrd.state_dict(),
                "proj_opt": proj_opt.state_dict(),
                "vocoder_opt": vocoder_opt.state_dict(),
                "mpd_opt": mpd_opt.state_dict(),
                "mrd_opt": mrd_opt.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "vocos_dim": args.vocos_dim,
                    "vocos_layers": args.vocos_layers,
                }
            }
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(args.output_dir, "hubert2mel_vocos_best.pt"))
            print(f"  ✓ Saved best checkpoint (val_loss: {val_loss:.4f})")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'hubert2mel_vocos_best.pt')}")


if __name__ == "__main__":
    main()
