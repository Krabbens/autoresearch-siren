"""
Stage 2: Train BitVocos Vocoder (with GAN)

This stage trains the vocoder using:
1. Reconstruction loss (L1 on audio)
2. Multi-Period Discriminator (MPD)
3. Multi-Resolution Discriminator (MRD)

Usage: uv run train_stage2.py --stage1_ckpt checkpoints/stage1_best.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp

from ultra_low_bitrate_codec.models.bit_vocos import BitVocos


# ---------------------------------------------------------------------------
# Discriminators (for GAN training)
# ---------------------------------------------------------------------------

class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator for GAN training."""
    
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
        # Reshape: (B, T) -> (B, C, T/p, p)
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
    """Multi-Resolution Discriminator."""
    
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
        # STFT
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            self.window, return_complex=True
        )
        spec = torch.view_as_real(spec)  # (B, F, T, 2)
        spec = spec.permute(0, 3, 1, 2)  # (B, 2, F, T)
        
        for conv in self.convs:
            spec = self.act(conv(spec))
        
        spec = self.conv_post(spec)
        spec = spec.flatten(1, -1)
        
        return spec


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def discriminator_loss(scores_real, scores_fake):
    """Hinge loss for discriminators."""
    loss_real = 0.0
    loss_fake = 0.0
    
    for real, fake in zip(scores_real, scores_fake):
        loss_real += torch.mean(F.relu(1.0 - real))
        loss_fake += torch.mean(F.relu(1.0 + fake))
    
    return loss_real + loss_fake


def generator_loss(scores_fake):
    """Hinge loss for generator."""
    loss = 0.0
    for fake in scores_fake:
        loss += torch.mean(F.relu(1.0 - fake))
    return loss


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Vocoder GAN Training")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/vqvae_autoresearch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vocos_dim", type=int, default=512)
    parser.add_argument("--vocos_layers", type=int, default=8)
    
    # GAN hyperparameters
    parser.add_argument("--gan_start_epoch", type=int, default=5)  # Start GAN after reconstruction warmup
    parser.add_argument("--mpd_weight", type=float, default=1.0)
    parser.add_argument("--mrd_weight", type=float, default=1.0)
    parser.add_argument("--feat_match_weight", type=float, default=10.0)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 2 Training: BitVocos Vocoder with GAN")
    print(f"  Device: {device}")
    
    # Load stage 1 checkpoint
    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    config = ckpt.get("config", {})
    
    print(f"\nLoaded stage 1 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  Stage 1 val mel loss: {ckpt.get('val_mel_loss', 'N/A')}")
    
    # Import models from stage 1
    from train_stage1 import BranchEncoder, BranchDecoder, VectorQuantizer, HuBERT2Mel
    from ultra_low_bitrate_codec.models.bithubert import BitHuBERT
    from prepare import DEFAULT_DATA_DIR, DEFAULT_VAL_DATA_DIR, DEFAULT_HUBERT_CKPT, make_dataloader, split_train_val_files, verify_assets
    
    data_dir = os.path.abspath(DEFAULT_DATA_DIR)
    hubert_ckpt = os.path.abspath(DEFAULT_HUBERT_CKPT)
    val_dir = DEFAULT_VAL_DATA_DIR.strip() if DEFAULT_VAL_DATA_DIR else ""
    train_paths, val_paths = split_train_val_files(data_dir, val_data_dir=val_dir)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pin = device.type == "cuda"
    train_loader = make_dataloader(
        train_paths, args.batch_size, shuffle=True, num_workers=4,
        prefetch_factor=2, pin_memory=pin, seed=int(time.time()) % (2**31)
    )

    # Load frozen models from stage 1
    hubert = BitHuBERT(hidden_dim=384, output_dim=768, num_layers=12).to(device).eval()
    hubert.load_state_dict(torch.load(hubert_ckpt, map_location=device))

    encoder = BranchEncoder(
        input_dim=768, hidden_dim=config.get('hidden_dim', 512),
        sem_dim=config.get('sem_dim', 128), pro_dim=config.get('pro_dim', 64), 
        spk_dim=config.get('spk_dim', 256),
        sem_compression=config.get('sem_compression', 2),
        pro_compression=config.get('pro_compression', 8),
        spk_compression=config.get('spk_compression', 2),
    ).to(device).eval()
    
    decoder = BranchDecoder(
        sem_dim=config.get('sem_dim', 128), pro_dim=config.get('pro_dim', 64), 
        spk_dim=config.get('spk_dim', 256),
        hidden_dim=config.get('hidden_dim', 512), output_dim=768,
        sem_compression=config.get('sem_compression', 2),
        pro_compression=config.get('pro_compression', 8),
        spk_compression=config.get('spk_compression', 2),
    ).to(device).eval()
    
    sem_vq = VectorQuantizer(
        num_codes=config.get('sem_num_codes', 32), 
        codebook_dim=config.get('sem_dim', 128),
        num_residuals=config.get('sem_num_residuals', 3)
    ).to(device).eval()
    
    pro_vq = VectorQuantizer(
        num_codes=config.get('pro_num_codes', 64), 
        codebook_dim=config.get('pro_dim', 64),
        num_residuals=config.get('pro_num_residuals', 1)
    ).to(device).eval()
    
    spk_vq = VectorQuantizer(
        num_codes=config.get('spk_num_codes', 32), 
        codebook_dim=config.get('spk_dim', 256),
        num_residuals=config.get('spk_num_residuals', 2)
    ).to(device).eval()
    
    hubert2mel = HuBERT2Mel(
        hubert_dim=768, mel_dim=80, hidden_dim=config.get('mel_hidden_dim', 512)
    ).to(device).eval()
    
    # Load weights
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    sem_vq.load_state_dict(ckpt["sem_vq"])
    pro_vq.load_state_dict(ckpt["pro_vq"])
    spk_vq.load_state_dict(ckpt["spk_vq"])
    hubert2mel.load_state_dict(ckpt["hubert2mel"])
    
    # Vocoder (trainable)
    vocoder = BitVocos(
        input_dim=80, dim=args.vocos_dim, num_layers=args.vocos_layers,
        n_fft=1024, hop_length=320
    ).to(device).train()
    
    # Discriminators
    mpd = MultiPeriodDiscriminator().to(device).train()
    mrd = MultiResolutionDiscriminator().to(device).train()
    
    # Optimizers
    vocoder_opt = torch.optim.AdamW(vocoder.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mpd_opt = torch.optim.AdamW(mpd.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mrd_opt = torch.optim.AdamW(mrd.parameters(), lr=args.lr, betas=(0.8, 0.99))
    
    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    for epoch in range(args.epochs):
        vocoder.train()
        
        # Only train discriminators after warmup
        train_gan = epoch >= args.gan_start_epoch
        
        if train_gan:
            mpd.train()
            mrd.train()
        else:
            mpd.eval()
            mrd.eval()
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_gan = 0.0
        epoch_feat = 0.0
        num_batches = 0
        
        for wav in train_loader:
            wav = wav.to(device, non_blocking=True)
            
            with torch.no_grad():
                wav_16k = wav.unsqueeze(1)
                h_feats, _ = hubert(wav_16k)
                
                sem, pro, spk = encoder(h_feats)
                sem_q, _, _ = sem_vq(sem)
                pro_q, _, _ = pro_vq(pro)
                spk_q, _, _ = spk_vq(spk)
                
                h_recon = decoder(sem_q, pro_q, spk_q, target_len=h_feats.shape[1])
                
                mel_pred = hubert2mel(h_recon).transpose(1, 2)
                mel_pred = torch.clamp(mel_pred, min=1e-5)
                mel_pred = torch.exp(mel_pred)  # Inverse log
            
            # Vocoder forward
            audio_pred = vocoder(mel_pred)
            
            # Align lengths
            target_len = wav.shape[1]
            if audio_pred.shape[1] > target_len:
                audio_pred = audio_pred[:, :target_len]
            elif audio_pred.shape[1] < target_len:
                audio_pred = F.pad(audio_pred, (0, target_len - audio_pred.shape[1]))
            
            # Reconstruction loss (L1 is better for audio)
            recon_loss = F.l1_loss(audio_pred, wav)
            
            # GAN losses
            gan_loss = torch.tensor(0.0, device=device)
            feat_loss = torch.tensor(0.0, device=device)
            
            if train_gan:
                # Train discriminators
                mpd_opt.zero_grad()
                mrd_opt.zero_grad()
                
                # Real scores
                scores_real_mpd = mpd(wav)
                scores_real_mrd = mrd(wav)
                
                # Fake scores (detach vocoder)
                scores_fake_mpd = mpd(audio_pred.detach())
                scores_fake_mrd = mrd(audio_pred.detach())
                
                # Discriminator losses
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
                
                # Generator losses
                scores_fake_mpd = mpd(audio_pred)
                scores_fake_mrd = mrd(audio_pred)
                
                gan_loss = (
                    args.mpd_weight * generator_loss(scores_fake_mpd) +
                    args.mrd_weight * generator_loss(scores_fake_mrd)
                )
                
                # Feature matching loss
                with torch.no_grad():
                    scores_real_mpd = mpd(wav)
                    scores_real_mrd = mrd(wav)
                
                for real, fake in zip(scores_real_mpd, scores_fake_mpd):
                    feat_loss += F.l1_loss(fake, real) * args.feat_match_weight
                for real, fake in zip(scores_real_mrd, scores_fake_mrd):
                    feat_loss += F.l1_loss(fake, real) * args.feat_match_weight
            
            # Total vocoder loss
            vocoder_loss = recon_loss + gan_loss + feat_loss
            
            vocoder_opt.zero_grad()
            
            if use_amp:
                scaler.scale(vocoder_loss).backward()
                scaler.unscale_(vocoder_opt)
            else:
                vocoder_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(vocoder.parameters(), 0.5)
            
            if use_amp:
                scaler.step(vocoder_opt)
                scaler.update()
            else:
                vocoder_opt.step()
            
            epoch_loss += vocoder_loss.item()
            epoch_recon += recon_loss.item()
            if train_gan:
                epoch_gan += gan_loss.item()
                epoch_feat += feat_loss.item()
            num_batches += 1
        
        # Print progress
        if train_gan:
            print(f"Epoch {epoch+1}/{args.epochs}: loss={epoch_loss/num_batches:.4f}, "
                  f"recon={epoch_recon/num_batches:.4f}, gan={epoch_gan/num_batches:.4f}, "
                  f"feat={epoch_feat/num_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}: loss={epoch_loss/num_batches:.4f}, "
                  f"recon={epoch_recon/num_batches:.4f} (warmup, no GAN)")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            full_ckpt = ckpt.copy()
            full_ckpt["vocoder"] = vocoder.state_dict()
            full_ckpt["mpd"] = mpd.state_dict()
            full_ckpt["mrd"] = mrd.state_dict()
            full_ckpt["vocoder_opt"] = vocoder_opt.state_dict()
            full_ckpt["mpd_opt"] = mpd_opt.state_dict()
            full_ckpt["mrd_opt"] = mrd_opt.state_dict()
            full_ckpt["epoch"] = epoch
            
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(full_ckpt, os.path.join(args.output_dir, "vqvae_vocos_stage2.pt"))
            print(f"  ✓ Saved checkpoint")
    
    print(f"\nStage 2 complete!")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'vqvae_vocos_stage2.pt')}")


if __name__ == "__main__":
    main()
