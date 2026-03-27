"""
Evaluate supervised HuBERT2Mel + BitVocos

Usage: uv run eval_hubert2mel_supervised.py
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


class HuBERT2Mel(torch.nn.Module):
    def __init__(self, hubert_dim=768, mel_dim=80, hidden_dim=1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hubert_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, mel_dim),
        )
        
    def forward(self, h):
        return self.net(h)


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
    parser.add_argument("--hubert2mel_ckpt", type=str, default="checkpoints/vqvae_autoresearch/hubert2mel_supervised_best.pt")
    parser.add_argument("--vocos_ckpt", type=str, default="/home/sperm/siren/SIREN/checkpoints/bitvocos_v8_16k/vocos_ep299.pt")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--val_data_dir", default=DEFAULT_VAL_DATA_DIR)
    parser.add_argument("--hubert_ckpt", type=str, default=DEFAULT_HUBERT_CKPT)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HuBERT2Mel (Supervised) + BitVocos Evaluation")
    print(f"  HuBERT2Mel: {args.hubert2mel_ckpt}")
    print(f"  BitVocos: {args.vocos_ckpt}")
    
    # Load BitVocos
    vocoder_ckpt = torch.load(args.vocos_ckpt, map_location=device)
    vocoder = BitVocos(input_dim=80, dim=512, num_layers=8, n_fft=1024, hop_length=320).to(device)
    if isinstance(vocoder_ckpt, dict) and 'model' in vocoder_ckpt:
        vocoder.load_state_dict(vocoder_ckpt['model'])
    elif isinstance(vocoder_ckpt, dict):
        vocoder.load_state_dict(vocoder_ckpt)
    vocoder.eval()
    print(f"  BitVocos loaded")
    
    # Load HuBERT2Mel
    if os.path.exists(args.hubert2mel_ckpt):
        hubert2mel_ckpt = torch.load(args.hubert2mel_ckpt, map_location=device)
        hubert2mel = HuBERT2Mel(hubert_dim=768, mel_dim=80, hidden_dim=1024).to(device)
        hubert2mel.load_state_dict(hubert2mel_ckpt['hubert2mel'])
        hubert2mel.eval()
        print(f"  HuBERT2Mel loaded (val_mse: {hubert2mel_ckpt.get('val_loss', 'N/A')})")
    else:
        print(f"  Warning: HuBERT2Mel checkpoint not found: {args.hubert2mel_ckpt}")
        return
    
    # Setup
    data_dir = os.path.abspath(args.data_dir)
    hubert_ckpt_path = os.path.abspath(args.hubert_ckpt)
    verify_assets(data_dir, hubert_ckpt_path, hubert_ckpt_path)

    val_dir = args.val_data_dir.strip() if args.val_data_dir else ""
    train_paths, val_paths = split_train_val_files(data_dir, val_data_dir=val_dir)
    eval_paths = val_paths[:args.num_samples]
    
    # Load HubERT
    hubert = BitHuBERT(hidden_dim=384, output_dim=768, num_layers=12).to(device).eval()
    hubert.load_state_dict(torch.load(hubert_ckpt_path, map_location=device))
    
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
            # HuBERT features
            h_feats, _ = hubert(wav_b)
            
            # HuBERT → Mel (supervised)
            mel_pred = hubert2mel(h_feats).transpose(1, 2)
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
