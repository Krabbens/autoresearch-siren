"""
Evaluate trained VQ-VAE + Vocoder

Usage: uv run eval_vocoder.py
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

from prepare import (
    DEFAULT_DATA_DIR,
    DEFAULT_HUBERT_CKPT,
    DEFAULT_VAL_DATA_DIR,
    split_train_val_files,
    verify_assets,
)
from ultra_low_bitrate_codec.models.bithubert import BitHuBERT
from ultra_low_bitrate_codec.models.bit_vocos import BitVocos


# Import models from train_vocoder.py
class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_codes, codebook_dim, num_residuals=1):
        super().__init__()
        self.num_codes = num_codes
        self.num_residuals = num_residuals
        
        if num_residuals == 1:
            self.embedding = torch.nn.Embedding(num_codes, codebook_dim)
        else:
            self.embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(num_codes, codebook_dim) for _ in range(num_residuals)
            ])
        
        # Buffers for loading checkpoints
        self.register_buffer("ema_count", torch.ones(num_codes))
        self.register_buffer("ema_weight", torch.ones((num_codes, codebook_dim)))
        
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
            return z_q, torch.stack(all_indices, dim=-1)


class BranchEncoder(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, 
                 sem_dim=128, pro_dim=64, spk_dim=256,
                 sem_compression=1, pro_compression=4, spk_compression=1):
        super().__init__()
        self.sem_compression = sem_compression
        self.pro_compression = pro_compression
        self.spk_compression = spk_compression
        
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
        sem = self._compress(self.semantic(h), self.sem_compression)
        pro = self._compress(self.prosody(h), self.pro_compression)
        spk = self._compress(self.speaker_encoder(x), self.spk_compression)
        return sem, pro, spk


class BranchDecoder(torch.nn.Module):
    def __init__(self, sem_dim=128, pro_dim=64, spk_dim=256, 
                 hidden_dim=512, output_dim=768,
                 sem_compression=1, pro_compression=4, spk_compression=1):
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


class HuBERT2Mel(torch.nn.Module):
    def __init__(self, hubert_dim=768, mel_dim=80, hidden_dim=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hubert_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, mel_dim),
        )
        
    def forward(self, h_recon):
        return self.net(h_recon)


def evaluate_pesq_stoi(ref_wav, deg_wav, sr=16000):
    try:
        from pesq import pesq as pesq_fn
        from pystoi import stoi as stoi_fn
    except ImportError:
        return float('nan'), float('nan')
    
    ref_np = ref_wav.detach().cpu().numpy().astype(np.float64)
    deg_np = deg_wav.detach().cpu().numpy().astype(np.float64)
    
    min_len = min(len(ref_np), len(deg_np))
    ref_np = ref_np[:min_len]
    deg_np = deg_np[:min_len]
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vqvae_autoresearch/vqvae_vocoder_best.pt")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"VQ-VAE + Vocoder Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"  Error: Checkpoint not found: {args.checkpoint}")
        return
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    print(f"  Config: {config}")
    
    # Load models
    hubert = BitHuBERT(hidden_dim=384, output_dim=768, num_layers=12).to(device).eval()
    hubert.load_state_dict(torch.load(args.hubert_ckpt, map_location=device))

    encoder = BranchEncoder(
        input_dim=768, hidden_dim=512,
        sem_dim=128, pro_dim=64, spk_dim=256,
        pro_compression=config.get('pro_compression', 4),
    ).to(device).eval()
    
    decoder = BranchDecoder(
        sem_dim=128, pro_dim=64, spk_dim=256,
        hidden_dim=512, output_dim=768,
        pro_compression=config.get('pro_compression', 4),
    ).to(device).eval()
    
    num_codes = config.get('num_codes', 1024)
    sem_vq = VectorQuantizer(num_codes=num_codes, codebook_dim=128).to(device).eval()
    pro_vq = VectorQuantizer(num_codes=num_codes, codebook_dim=64).to(device).eval()
    spk_vq = VectorQuantizer(num_codes=num_codes, codebook_dim=256).to(device).eval()
    
    hubert2mel = HuBERT2Mel(hubert_dim=768, mel_dim=80, hidden_dim=512).to(device).eval()
    
    vocoder = BitVocos(
        input_dim=80, 
        dim=config.get('vocos_dim', 384), 
        num_layers=config.get('vocos_layers', 6),
        n_fft=1024, hop_length=320
    ).to(device).eval()
    
    # Load weights (non-strict to handle buffer mismatches)
    encoder.load_state_dict(ckpt["encoder"], strict=False)
    decoder.load_state_dict(ckpt["decoder"], strict=False)
    sem_vq.load_state_dict(ckpt["sem_vq"], strict=False)
    pro_vq.load_state_dict(ckpt["pro_vq"], strict=False)
    spk_vq.load_state_dict(ckpt["spk_vq"], strict=False)
    hubert2mel.load_state_dict(ckpt["hubert2mel"], strict=False)
    vocoder.load_state_dict(ckpt["vocoder"], strict=False)
    
    # Setup data
    data_dir = os.path.abspath(args.data_dir)
    hubert_ckpt_path = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, hubert_ckpt_path, hubert_ckpt_path)

    val_dir = args.val_data_dir.strip() if args.val_data_dir else ""
    train_paths, val_paths = split_train_val_files(data_dir, val_data_dir=val_dir)
    eval_paths = val_paths[:args.num_samples]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pesq_scores = []
    stoi_scores = []
    
    print(f"\nEvaluating {len(eval_paths)} samples...")
    print("=" * 70)
    print(f"{'File':<45} {'PESQ':>8} {'STOI':>8}")
    print("-" * 70)
    
    for wav_path in tqdm(eval_paths, desc="Evaluating"):
        wav_np, sr = sf.read(wav_path)
        
        if sr != 16000:
            import torchaudio
            wav_np = torchaudio.functional.resample(
                torch.tensor(wav_np).unsqueeze(0), sr, 16000
            ).squeeze(0).numpy()
            sr = 16000
        
        wav_t = torch.tensor(wav_np, dtype=torch.float32).to(device)
        if wav_t.shape[0] > 4 * 16000:
            wav_t = wav_t[:4 * 16000]
        wav_b = wav_t.unsqueeze(0).unsqueeze(1)
        
        with torch.no_grad():
            # VQ-VAE
            h_feats, _ = hubert(wav_b)
            sem, pro, spk = encoder(h_feats)
            sem_q, _ = sem_vq(sem)
            pro_q, _ = pro_vq(pro)
            spk_q, _ = spk_vq(spk)
            h_recon = decoder(sem_q, pro_q, spk_q, target_len=h_feats.shape[1])
            
            # Mel prediction
            mel_pred = hubert2mel(h_recon).transpose(1, 2)
            mel_pred = torch.clamp(mel_pred, min=1e-5)
            mel_pred = torch.exp(mel_pred)
            
            # Vocoder
            audio_pred = vocoder(mel_pred)
            
            ref_audio = wav_b.squeeze()
            if audio_pred.shape[1] > ref_audio.shape[0]:
                audio_pred = audio_pred[:, :ref_audio.shape[0]]
            elif audio_pred.shape[1] < ref_audio.shape[0]:
                ref_audio = ref_audio[:audio_pred.shape[1]]
            
            pesq, stoi = evaluate_pesq_stoi(ref_audio, audio_pred.squeeze(), sr=16000)
        
        if math.isfinite(pesq):
            pesq_scores.append(pesq)
        if math.isfinite(stoi):
            stoi_scores.append(stoi)
        
        filename = os.path.basename(wav_path)[:43]
        pesq_str = f"{pesq:.4f}" if math.isfinite(pesq) else "nan"
        stoi_str = f"{stoi:.4f}" if math.isfinite(stoi) else "nan"
        print(f"{filename:<45} {pesq_str:>8} {stoi_str:>8}")
        
        # Save audio
        sample_name = os.path.basename(wav_path).replace('.wav', '_recon.wav')
        output_path = os.path.join(args.output_dir, sample_name)
        sf.write(output_path, audio_pred.squeeze().cpu().numpy(), 16000)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if pesq_scores:
        print(f"PESQ (wideband):  {np.mean(pesq_scores):.4f} ± {np.std(pesq_scores):.4f}")
        print(f"  Min: {np.min(pesq_scores):.4f}, Max: {np.max(pesq_scores):.4f}")
        quality = 'Excellent' if np.mean(pesq_scores) >= 4.0 else 'Good' if np.mean(pesq_scores) >= 3.0 else 'Fair' if np.mean(pesq_scores) >= 2.0 else 'Poor'
        print(f"  Quality: {quality}")
    else:
        print("PESQ: N/A")
    
    if stoi_scores:
        print(f"STOI:             {np.mean(stoi_scores):.4f} ± {np.std(stoi_scores):.4f}")
    else:
        print("STOI: N/A")
    
    print(f"\nAudio samples saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
