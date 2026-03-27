"""
Train BitVocos Vocoder with VQ-VAE

End-to-end audio codec training:
- VQ-VAE compresses HuBERT features
- HuBERT2Mel projects to mel spectrogram  
- BitVocos synthesizes audio from mel

Uses GAN training with MPD + MRD discriminators.

Usage: uv run train_vocoder.py --epochs 100
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
from ultra_low_bitrate_codec.models.bit_vocos import BitVocos


# ---------------------------------------------------------------------------
# VQ-VAE Models (matching train.py)
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
        sem = self._compress(self.semantic(h), self.sem_compression)
        pro = self._compress(self.prosody(h), self.pro_compression)
        spk = self._compress(self.speaker_encoder(x), self.spk_compression)
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
# Discriminators (for GAN training)
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


# ---------------------------------------------------------------------------
# Loss Functions
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


def feature_matching_loss(features_real, features_fake, weight=10.0):
    """Feature matching loss."""
    loss = 0.0
    for real, fake in zip(features_real, features_fake):
        loss += F.l1_loss(fake, real.detach()) * weight
    return loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE + Vocoder")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    
    # VQ-VAE config
    parser.add_argument("--num_codes", type=int, default=1024)
    parser.add_argument("--pro_compression", type=int, default=4)
    parser.add_argument("--commitment_beta", type=float, default=1.0)
    
    # Vocoder config - optimized for speed
    parser.add_argument("--vocos_dim", type=int, default=256)
    parser.add_argument("--vocos_layers", type=int, default=4)
    
    # GAN config - optimized for speed
    parser.add_argument("--gan_start_epoch", type=int, default=5)
    parser.add_argument("--mpd_weight", type=float, default=1.0)
    parser.add_argument("--mrd_weight", type=float, default=1.0)
    parser.add_argument("--feat_match_weight", type=float, default=10.0)
    parser.add_argument("--mel_weight", type=float, default=45.0)
    parser.add_argument("--audio_weight", type=float, default=1.0)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"VQ-VAE + Vocoder Training")
    print(f"  Device: {device}")
    print(f"  Bitrate: ~{10 * math.log2(args.num_codes) * (50 + 50/args.pro_compression + 50):.0f} bps")
    
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
        input_dim=768, hidden_dim=512,
        sem_dim=128, pro_dim=64, spk_dim=256,
        pro_compression=args.pro_compression,
    ).to(device)
    
    decoder = BranchDecoder(
        sem_dim=128, pro_dim=64, spk_dim=256,
        hidden_dim=512, output_dim=768,
        pro_compression=args.pro_compression,
    ).to(device)
    
    sem_vq = VectorQuantizer(
        num_codes=args.num_codes, codebook_dim=128,
        beta=args.commitment_beta, num_residuals=1
    ).to(device)
    
    pro_vq = VectorQuantizer(
        num_codes=args.num_codes, codebook_dim=64,
        beta=args.commitment_beta, num_residuals=1
    ).to(device)
    
    spk_vq = VectorQuantizer(
        num_codes=args.num_codes, codebook_dim=256,
        beta=args.commitment_beta, num_residuals=1
    ).to(device)
    
    hubert2mel = HuBERT2Mel(hubert_dim=768, mel_dim=80, hidden_dim=512).to(device)
    
    vocoder = BitVocos(
        input_dim=80, dim=args.vocos_dim, num_layers=args.vocos_layers,
        n_fft=1024, hop_length=320
    ).to(device)
    
    # Discriminators
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    
    # Optimizers
    vqvae_params = (list(encoder.parameters()) + list(decoder.parameters()) + 
                    list(sem_vq.parameters()) + list(pro_vq.parameters()) + 
                    list(spk_vq.parameters()) + list(hubert2mel.parameters()))
    vqvae_opt = torch.optim.AdamW(vqvae_params, lr=args.lr, betas=(0.8, 0.99))
    
    vocoder_opt = torch.optim.AdamW(vocoder.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mpd_opt = torch.optim.AdamW(mpd.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mrd_opt = torch.optim.AdamW(mrd.parameters(), lr=args.lr, betas=(0.8, 0.99))
    
    # Scheduler
    vqvae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vqvae_opt, T_max=args.epochs)
    vocoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vocoder_opt, T_max=args.epochs)
    
    # Mel spectrogram for ground truth
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80
    ).to(device)

    use_amp = device.type == "cuda"
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        sem_vq.train()
        pro_vq.train()
        spk_vq.train()
        hubert2mel.train()
        vocoder.train()
        
        # Only train GAN after warmup
        train_gan = epoch >= args.gan_start_epoch
        
        if train_gan:
            mpd.train()
            mrd.train()
        else:
            mpd.eval()
            mrd.eval()
        
        epoch_vqvae_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_audio_loss = 0.0
        epoch_gan_loss = 0.0
        num_batches = 0
        
        for wav in train_loader:
            wav = wav.to(device, non_blocking=True)
            
            # Ground truth mel and audio
            wav_16k = wav.unsqueeze(1)
            with torch.no_grad():
                mel_gt = mel_transform(wav_16k)
                mel_gt = torch.log(torch.clamp(mel_gt, min=1e-5))
                if mel_gt.dim() == 4:
                    mel_gt = mel_gt.squeeze(1)
                
                h_feats, _ = hubert(wav_16k)
            
            # VQ-VAE forward
            sem, pro, spk = encoder(h_feats)
            
            sem_q, sem_vq_loss, _ = sem_vq(sem)
            pro_q, pro_vq_loss, _ = pro_vq(pro)
            spk_q, spk_vq_loss, _ = spk_vq(spk)
            
            h_recon = decoder(sem_q, pro_q, spk_q, target_len=h_feats.shape[1])
            
            # Mel prediction
            mel_pred = hubert2mel(h_recon).transpose(1, 2)
            
            # Align lengths
            min_len = min(mel_pred.shape[2], mel_gt.shape[2])
            mel_pred = mel_pred[:, :, :min_len]
            mel_gt = mel_gt[:, :, :min_len]
            
            # Mel reconstruction loss
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
            
            # Audio reconstruction loss
            audio_loss = F.l1_loss(audio_pred, wav)
            
            # Total VQ loss
            total_vq_loss = sem_vq_loss + pro_vq_loss + spk_vq_loss
            
            # Generator GAN loss
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
                
                gan_loss = (
                    args.mpd_weight * generator_loss(scores_fake_mpd) +
                    args.mrd_weight * generator_loss(scores_fake_mrd)
                )
                
                # Feature matching
                with torch.no_grad():
                    scores_real_mpd = mpd(wav)
                    scores_real_mrd = mrd(wav)
                
                feat_loss = (
                    feature_matching_loss(scores_fake_mpd, scores_real_mpd, args.feat_match_weight) +
                    feature_matching_loss(scores_fake_mrd, scores_real_mrd, args.feat_match_weight)
                )
            
            # Total loss
            vqvae_loss = (
                F.mse_loss(h_recon, h_feats) +
                args.commitment_beta * total_vq_loss +
                args.mel_weight * mel_loss
            )
            
            total_loss = vqvae_loss + args.audio_weight * audio_loss + gan_loss + feat_loss
            
            # Optimize VQ-VAE + vocoder
            vqvae_opt.zero_grad()
            vocoder_opt.zero_grad()
            
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(vqvae_opt)
                scaler.unscale_(vocoder_opt)
            else:
                total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(vocoder.parameters()), 0.5)
            
            if use_amp:
                scaler.step(vqvae_opt)
                scaler.step(vocoder_opt)
                scaler.update()
            else:
                vqvae_opt.step()
                vocoder_opt.step()
            
            epoch_vqvae_loss += vqvae_loss.item()
            epoch_mel_loss += mel_loss.item()
            epoch_audio_loss += audio_loss.item()
            if train_gan:
                epoch_gan_loss += gan_loss.item()
            num_batches += 1
        
        vqvae_scheduler.step()
        vocoder_scheduler.step()
        
        # Validation
        encoder.eval()
        decoder.eval()
        sem_vq.eval()
        pro_vq.eval()
        spk_vq.eval()
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
        train_vqvae_loss = epoch_vqvae_loss / max(1, num_batches)
        train_mel_loss = epoch_mel_loss / max(1, num_batches)
        train_audio_loss = epoch_audio_loss / max(1, num_batches)
        
        if train_gan:
            train_gan = epoch_gan_loss / max(1, num_batches)
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"vqvae={train_vqvae_loss:.4f}, mel={train_mel_loss:.4f}, "
                  f"audio={train_audio_loss:.4f}, gan={train_gan:.4f}, "
                  f"val_mel={val_mel_loss:.4f}, val_audio={val_audio_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"vqvae={train_vqvae_loss:.4f}, mel={train_mel_loss:.4f}, "
                  f"audio={train_audio_loss:.4f} (warmup), "
                  f"val_mel={val_mel_loss:.4f}, val_audio={val_audio_loss:.4f}")
        
        # Save checkpoint
        val_loss = val_mel_loss + val_audio_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "sem_vq": sem_vq.state_dict(),
                "pro_vq": pro_vq.state_dict(),
                "spk_vq": spk_vq.state_dict(),
                "hubert2mel": hubert2mel.state_dict(),
                "vocoder": vocoder.state_dict(),
                "mpd": mpd.state_dict(),
                "mrd": mrd.state_dict(),
                "vqvae_opt": vqvae_opt.state_dict(),
                "vocoder_opt": vocoder_opt.state_dict(),
                "mpd_opt": mpd_opt.state_dict(),
                "mrd_opt": mrd_opt.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "num_codes": args.num_codes,
                    "pro_compression": args.pro_compression,
                    "vocos_dim": args.vocos_dim,
                    "vocos_layers": args.vocos_layers,
                }
            }
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(args.output_dir, "vqvae_vocoder_best.pt"))
            print(f"  ✓ Saved best checkpoint (val_loss: {val_loss:.4f})")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'vqvae_vocoder_best.pt')}")


if __name__ == "__main__":
    main()
