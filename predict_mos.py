#!/usr/bin/env python3
"""
SingMOS Prediction Script
Use a trained model to predict MOS scores for audio files.
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import soundfile as sf
from train_singmos import MERTEncoder, SingMOSModel, mean_std_pooling


def get_args():
    parser = argparse.ArgumentParser(description="Predict MOS scores for audio files")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file or directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--model_name", type=str, default="m-a-p/MERT-v1-95M", help="MERT model name")
    parser.add_argument("--mos_mean", type=float, default=None, help="MOS mean (if not in checkpoint)")
    parser.add_argument("--mos_std", type=float, default=None, help="MOS std (if not in checkpoint)")
    return parser.parse_args()


def load_audio(path, target_sr=16000):
    """Load and preprocess audio file"""
    wav, sr = sf.read(path, dtype="float32")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = torch.from_numpy(wav)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav


def predict_single(model, wav, device, MOS_MEAN, MOS_STD):
    """Predict MOS for a single audio file"""
    model.eval()

    # Create mask (all valid)
    length = wav.shape[0]
    mask = torch.ones(1, length, dtype=torch.bool)

    # Add batch dimension
    wav = wav.unsqueeze(0).to(device)
    mask = mask.to(device)

    with torch.no_grad():
        z_pred = model(wav, mask)
        mos_pred = (z_pred * MOS_STD + MOS_MEAN).clamp(1.0, 5.0)

    return mos_pred.item()


def load_model_and_stats(ckpt_path, model_name, device, mos_mean=None, mos_std=None):
    """Load model and statistics from checkpoint"""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Build model
    encoder = MERTEncoder(model_name=model_name, device=device)
    model = SingMOSModel(encoder).to(device)
    model.load_state_dict(ckpt["model"])

    # Get MOS stats from checkpoint if available
    if mos_mean is not None and mos_std is not None:
        MOS_MEAN = mos_mean
        MOS_STD = mos_std
    elif "mos_mean" in ckpt and "mos_std" in ckpt:
        MOS_MEAN = ckpt["mos_mean"]
        MOS_STD = ckpt["mos_std"]
    else:
        # Default values (should be computed from training data)
        print("Warning: MOS mean/std not found in checkpoint. Using defaults.")
        print("For accurate predictions, provide --mos_mean and --mos_std")
        MOS_MEAN = 3.5  # Approximate default
        MOS_STD = 0.8   # Approximate default

    print(f"MOS mean: {MOS_MEAN:.4f}, std: {MOS_STD:.4f}")
    return model, MOS_MEAN, MOS_STD


def main():
    args = get_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, MOS_MEAN, MOS_STD = load_model_and_stats(
        args.ckpt, args.model_name, device,
        args.mos_mean, args.mos_std
    )

    # Process audio
    audio_path = args.audio

    if os.path.isfile(audio_path):
        # Single file
        print(f"\nProcessing: {audio_path}")
        wav = load_audio(audio_path)
        mos = predict_single(model, wav, device, MOS_MEAN, MOS_STD)
        print(f"Predicted MOS: {mos:.3f}")

    elif os.path.isdir(audio_path):
        # Directory
        audio_files = []
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            audio_files.extend([
                os.path.join(audio_path, f)
                for f in os.listdir(audio_path)
                if f.lower().endswith(ext)
            ])

        if not audio_files:
            print(f"No audio files found in {audio_path}")
            sys.exit(1)

        print(f"\nFound {len(audio_files)} audio files")
        print(f"{'File':<50} {'MOS Score':>10}")
        print("-" * 65)

        results = []
        for filepath in sorted(audio_files):
            try:
                wav = load_audio(filepath)
                mos = predict_single(model, wav, device, MOS_MEAN, MOS_STD)
                filename = os.path.basename(filepath)
                print(f"{filename:<50} {mos:>10.3f}")
                results.append((filepath, mos))
            except Exception as e:
                print(f"{os.path.basename(filepath):<50} {'ERROR':>10} ({e})")

        # Summary
        if results:
            scores = [mos for _, mos in results]
            print("-" * 65)
            print(f"{'Average':<50} {np.mean(scores):>10.3f}")
            print(f"{'Std Dev':<50} {np.std(scores):>10.3f}")
            print(f"{'Min':<50} {np.min(scores):>10.3f}")
            print(f"{'Max':<50} {np.max(scores):>10.3f}")

    else:
        print(f"Error: {audio_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
