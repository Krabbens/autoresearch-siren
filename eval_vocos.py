"""
Evaluate VQ-VAE + BitVocos model with real audio metrics (PESQ, STOI)

Usage: uv run eval_vocos.py --checkpoint checkpoints/vqvae_autoresearch/vqvae_vocos_last.pt
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

from prepare import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_HUBERT_CKPT,
    DEFAULT_VAL_DATA_DIR,
    make_dataloader,
    split_train_val_files,
    verify_assets,
)
from ultra_low_bitrate_codec.models.bithubert import BitHuBERT


# ---------------------------------------------------------------------------
# Model definitions (must match train_vocos.py)
# ---------------------------------------------------------------------------

class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_codes, codebook_dim, beta=0.25, num_residuals=1):
        super().__init__()
        self.num_codes = num_codes
        self.codebook_dim = codebook_dim
        self.num_residuals = num_residuals
        
        if num_residuals == 1:
            self.embedding = torch.nn.Embedding(num_codes, codebook_dim)
        else:
            self.embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(num_codes, codebook_dim) for _ in range(num_residuals)
            ])
        
        # Buffers for codebook usage tracking (for saving/loading)
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
            z_q = z + (z_q - z).detach()
            return z_q, indices
        else:
            residual = z_flat
            z_q_flat = torch.zeros_like(residual)
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
                all_indices.append(indices)
            
            z_q = z_q_flat.reshape(z.shape)
            z_q = z + (z_q_flat.reshape(z.shape) - z).detach()
            indices = torch.stack(all_indices, dim=-1)
            return z_q, indices


class BranchEncoder(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, 
                 sem_dim=128, pro_dim=64, spk_dim=256,
                 sem_compression=1, pro_compression=4, spk_compression=1,
                 speaker_mode="temporal"):
        super().__init__()
        self.sem_compression = sem_compression
        self.pro_compression = pro_compression
        self.spk_compression = spk_compression
        self.speaker_mode = speaker_mode
        
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )
        
        self.semantic = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, sem_dim),
        )
        
        self.prosody = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, pro_dim),
        )
        
        if speaker_mode == "global":
            self.speaker_encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, spk_dim),
            )
            self.speaker_attn = torch.nn.Linear(hidden_dim, 1)
        else:
            self.speaker_encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, spk_dim),
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


class BranchDecoder(torch.nn.Module):
    def __init__(self, sem_dim=128, pro_dim=64, spk_dim=256, 
                 hidden_dim=512, output_dim=768,
                 sem_compression=1, pro_compression=4, spk_compression=1,
                 speaker_mode="temporal"):
        super().__init__()
        self.sem_compression = sem_compression
        self.pro_compression = pro_compression
        self.spk_compression = spk_compression
        total_dim = sem_dim + pro_dim + spk_dim
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(total_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
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


class HuBERT2Mel(torch.nn.Module):
    def __init__(self, hubert_dim=768, mel_dim=80):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hubert_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Linear(512, mel_dim),
        )
        
    def forward(self, h_recon):
        return self.net(h_recon)


# Import BitVocos from SIREN
from ultra_low_bitrate_codec.models.bit_vocos import BitVocos


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pesq_stoi(ref_wav, deg_wav, sr=16000):
    """Calculate PESQ and STOI between reference and degraded audio."""
    try:
        from pesq import pesq as pesq_fn
        from pystoi import stoi as stoi_fn
    except ImportError:
        return float('nan'), float('nan')
    
    ref_np = ref_wav.detach().cpu().numpy().astype(np.float64)
    deg_np = deg_wav.detach().cpu().numpy().astype(np.float64)
    
    # Ensure same length
    min_len = min(len(ref_np), len(deg_np))
    ref_np = ref_np[:min_len]
    deg_np = deg_np[:min_len]
    
    # Normalize
    ref_np = ref_np / (np.max(np.abs(ref_np)) + 1e-8)
    deg_np = deg_np / (np.max(np.abs(deg_np)) + 1e-8)
    
    pesq = float('nan')
    stoi = float('nan')
    
    try:
        pesq = pesq_fn(sr, ref_np, deg_np, 'wb')
    except Exception:
        pass
    
    try:
        stoi = stoi_fn(ref_np, deg_np, sr, extended=False)
    except Exception:
        pass
    
    return pesq, stoi


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate VQ-VAE + BitVocos")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    
    print(f"\nLoaded config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Setup paths
    data_dir = os.path.abspath(args.data_dir)
    config_path = os.path.abspath(args.config)
    hubert_ckpt = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, config_path, hubert_ckpt)

    val_dir = args.val_data_dir.strip() if args.val_data_dir else ""
    train_paths, val_paths = split_train_val_files(data_dir, val_data_dir=val_dir)
    
    # Use subset for evaluation
    eval_paths = val_paths[:args.num_samples]
    print(f"\nEvaluating on {len(eval_paths)} samples")

    # Load models
    hubert = BitHuBERT(hidden_dim=384, output_dim=768, num_layers=12).to(device).eval()
    hubert.load_state_dict(torch.load(hubert_ckpt, map_location=device))

    encoder = BranchEncoder(
        input_dim=768, hidden_dim=config.get('hidden_dim', 512),
        sem_dim=config.get('sem_dim', 128), pro_dim=config.get('pro_dim', 64), 
        spk_dim=config.get('spk_dim', 256),
        sem_compression=config.get('sem_compression', 2),
        pro_compression=config.get('pro_compression', 8),
        spk_compression=config.get('spk_compression', 2),
        speaker_mode=config.get('speaker_mode', 'temporal')
    ).to(device)
    
    decoder = BranchDecoder(
        sem_dim=config.get('sem_dim', 128), pro_dim=config.get('pro_dim', 64), 
        spk_dim=config.get('spk_dim', 256),
        hidden_dim=config.get('hidden_dim', 512), output_dim=768,
        sem_compression=config.get('sem_compression', 2),
        pro_compression=config.get('pro_compression', 8),
        spk_compression=config.get('spk_compression', 2),
        speaker_mode=config.get('speaker_mode', 'temporal')
    ).to(device)
    
    sem_vq = VectorQuantizer(
        num_codes=config.get('sem_num_codes', 32), 
        codebook_dim=config.get('sem_dim', 128),
        num_residuals=config.get('sem_num_residuals', 3)
    ).to(device)
    
    pro_vq = VectorQuantizer(
        num_codes=config.get('pro_num_codes', 64), 
        codebook_dim=config.get('pro_dim', 64),
        num_residuals=config.get('pro_num_residuals', 1)
    ).to(device)
    
    spk_vq = VectorQuantizer(
        num_codes=config.get('spk_num_codes', 32), 
        codebook_dim=config.get('spk_dim', 256),
        num_residuals=config.get('spk_num_residuals', 2)
    ).to(device)
    
    hubert2mel = HuBERT2Mel(hubert_dim=768, mel_dim=80).to(device)
    
    vocoder = BitVocos(
        input_dim=80, 
        dim=config.get('vocos_dim', 256), 
        num_layers=config.get('vocos_layers', 4),
        n_fft=1024, hop_length=320
    ).to(device)
    
    # Load weights
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    sem_vq.load_state_dict(ckpt["sem_vq"])
    pro_vq.load_state_dict(ckpt["pro_vq"])
    spk_vq.load_state_dict(ckpt["spk_vq"])
    hubert2mel.load_state_dict(ckpt["hubert2mel"])
    vocoder.load_state_dict(ckpt["vocoder"])
    
    encoder.eval()
    decoder.eval()
    sem_vq.eval()
    pro_vq.eval()
    spk_vq.eval()
    hubert2mel.eval()
    vocoder.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluation metrics
    pesq_scores = []
    stoi_scores = []
    snr_scores = []
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'File':<40} {'PESQ':>8} {'STOI':>8} {'SNR(dB)':>10}")
    print("-" * 60)
    
    for wav_path in tqdm(eval_paths, desc="Evaluating"):
        # Load audio
        import soundfile as sf
        wav_np, sr = sf.read(wav_path)
        
        if sr != 16000:
            # Resample to 16kHz
            wav_np = torchaudio.functional.resample(
                torch.tensor(wav_np).unsqueeze(0), sr, 16000
            ).squeeze(0).numpy()
            sr = 16000
        
        # Prepare input
        wav_t = torch.tensor(wav_np, dtype=torch.float32).to(device)
        if wav_t.shape[0] > 4 * 16000:
            wav_t = wav_t[:4 * 16000]
        wav_b = wav_t.unsqueeze(0).unsqueeze(1)  # (B=1, 1, T)
        
        # Encode through HuBERT
        with torch.no_grad():
            h_feats, _ = hubert(wav_b)
            
            # VQ-VAE
            sem, pro, spk = encoder(h_feats)
            sem_q, _ = sem_vq(sem)
            pro_q, _ = pro_vq(pro)
            spk_q, _ = spk_vq(spk)
            
            # Decode HuBERT features
            h_recon = decoder(sem_q, pro_q, spk_q, target_len=h_feats.shape[1])
            
            # Project to mel
            mel_pred = hubert2mel(h_recon)  # (B, T, 80)
            mel_pred = mel_pred.transpose(1, 2)  # (B, 80, T)
            mel_pred = torch.clamp(mel_pred, min=1e-5)
            mel_pred = torch.exp(mel_pred)  # Inverse log
            
            # Synthesize audio
            audio_pred = vocoder(mel_pred)  # (B, T_audio)
            
            # Compute SNR
            ref_audio = wav_b.squeeze()
            if audio_pred.shape[1] > ref_audio.shape[0]:
                audio_pred = audio_pred[:, :ref_audio.shape[0]]
            elif audio_pred.shape[1] < ref_audio.shape[0]:
                ref_audio = ref_audio[:audio_pred.shape[1]]
            
            signal_power = torch.mean(ref_audio ** 2)
            noise_power = torch.mean((ref_audio - audio_pred.squeeze()) ** 2)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()
            
            # PESQ/STOI
            pesq, stoi = evaluate_pesq_stoi(ref_audio, audio_pred.squeeze(), sr=16000)
        
        # Store results
        if math.isfinite(pesq):
            pesq_scores.append(pesq)
        if math.isfinite(stoi):
            stoi_scores.append(stoi)
        if math.isfinite(snr):
            snr_scores.append(snr)
        
        # Print result
        filename = os.path.basename(wav_path)[:38]
        pesq_str = f"{pesq:.4f}" if math.isfinite(pesq) else "nan"
        stoi_str = f"{stoi:.4f}" if math.isfinite(stoi) else "nan"
        print(f"{filename:<40} {pesq_str:>8} {stoi_str:>8} {snr:>10.2f}")
        
        # Save audio sample using soundfile (more reliable than torchaudio)
        try:
            sample_name = os.path.basename(wav_path).replace('.wav', '_recon.wav')
            output_path = os.path.join(args.output_dir, sample_name)
            sf.write(output_path, audio_pred.squeeze().cpu().numpy(), 16000)
        except Exception as e:
            print(f"  Warning: Could not save audio: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if pesq_scores:
        print(f"PESQ (wideband):  {np.mean(pesq_scores):.4f} ± {np.std(pesq_scores):.4f}")
        print(f"  Min: {np.min(pesq_scores):.4f}, Max: {np.max(pesq_scores):.4f}")
    else:
        print("PESQ: N/A (pesq package not installed or evaluation failed)")
    
    if stoi_scores:
        print(f"STOI:             {np.mean(stoi_scores):.4f} ± {np.std(stoi_scores):.4f}")
        print(f"  Min: {np.min(stoi_scores):.4f}, Max: {np.max(stoi_scores):.4f}")
    else:
        print("STOI: N/A (pystoi package not installed or evaluation failed)")
    
    if snr_scores:
        print(f"SNR (dB):         {np.mean(snr_scores):.2f} ± {np.std(snr_scores):.2f}")
    
    print(f"\nAudio samples saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
