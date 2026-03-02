#!/usr/bin/env python3
"""
SingMOS Training Script
Train a MOS (Mean Opinion Score) prediction model for singing voice synthesis evaluation.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# ============================
# Configuration
# ============================

def get_args():
    parser = argparse.ArgumentParser(description="Train SingMOS model")
    parser.add_argument("--data_root", type=str, default="./SingMOS", help="Path to SingMOS dataset")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--model_name", type=str, default="m-a-p/MERT-v1-95M", help="MERT model name")
    parser.add_argument("--download_data", action="store_true", help="Download dataset from HuggingFace")
    return parser.parse_args()

# ============================
# Dataset
# ============================

class SingMOSDataset(Dataset):
    def __init__(self, items, sample_rate=16000):
        self.items = items
        self.sr = sample_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, mos = self.items[idx]
        wav, sr = sf.read(path, dtype="float32")

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        wav = torch.from_numpy(wav)

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        return wav, torch.tensor(mos, dtype=torch.float32)


def collate_fn(batch):
    wavs, mos = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in wavs])

    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    mask = torch.arange(wavs.size(1))[None, :] < lengths[:, None]
    return wavs, mask, torch.stack(mos)


def build_items(utt_ids, score_data, data_root):
    items = []
    for utt_id in utt_ids:
        info = score_data[utt_id]
        wav_path = os.path.join(data_root, info["wav"])
        mos = float(info["score"]["mos"])
        items.append((wav_path, mos))
    return items


def make_bin_weights(mos_list, bins=(1, 2, 3, 4, 5.01), max_ratio=10.0):
    mos = np.array(mos_list, dtype=np.float32)
    bin_idx = np.digitize(mos, bins) - 1
    n_bins = len(bins) - 1

    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)
    counts[counts == 0] = 1.0

    inv = 1.0 / counts
    inv = inv / inv.mean()
    inv = np.clip(inv, 1.0 / max_ratio, max_ratio)

    weights = inv[bin_idx]
    return torch.tensor(weights, dtype=torch.double), counts


# ============================
# Model Architecture
# ============================

class MERTEncoder(nn.Module):
    def __init__(self, model_name="m-a-p/MERT-v1-95M", layer_mode="last4", device=None):
        super().__init__()
        print(f"Loading MERT model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.layer_mode = layer_mode
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, wav, mask):
        dev = next(self.model.parameters()).device
        wav = wav.to(dev)
        mask = mask.to(dev)

        outputs = self.model(
            wav,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )

        if self.layer_mode == "last":
            h = outputs.last_hidden_state
        elif self.layer_mode == "last4":
            h = torch.stack(outputs.hidden_states[-4:]).mean(dim=0)
        else:
            raise ValueError(f"Unknown layer_mode: {self.layer_mode}")

        return h


def mean_std_pooling(h, frame_mask=None, eps=1e-6):
    if frame_mask is None:
        mean = h.mean(dim=1)
        std = h.std(dim=1)
        return torch.cat([mean, std], dim=-1)

    m = frame_mask.float().unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)
    mean = (h * m).sum(dim=1) / denom

    var = ((h - mean.unsqueeze(1)) ** 2) * m
    var = var.sum(dim=1) / denom
    std = torch.sqrt(var + eps)

    return torch.cat([mean, std], dim=-1)


class MOSRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SingMOSModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(2 * encoder.hidden_size)
        self.regressor = MOSRegressor(2 * encoder.hidden_size)

    def forward(self, wav, mask):
        h = self.encoder(wav, mask)
        B, T, _ = h.shape

        valid_samples = mask.sum(dim=1)
        L = mask.size(1)
        valid_frames = torch.ceil(valid_samples.float() * T / L).clamp(1, T).long()
        frame_mask = torch.arange(T, device=h.device)[None, :] < valid_frames[:, None]

        pooled = mean_std_pooling(h, frame_mask=frame_mask)
        pooled = self.norm(pooled)
        z_pred = self.regressor(pooled)
        return z_pred


# ============================
# Loss and Metrics
# ============================

def pearson_corr(x, y, eps=1e-8):
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (
        torch.sqrt((x * x).sum() + eps) *
        torch.sqrt((y * y).sum() + eps)
    )


def hybrid_loss(z_hat, y_true_mos, MOS_MEAN, MOS_STD, alpha=0.6):
    z_true = (y_true_mos - MOS_MEAN) / MOS_STD
    l1 = torch.nn.functional.l1_loss(z_hat, z_true)
    corr = pearson_corr(z_hat, z_true)
    return alpha * l1 + (1 - alpha) * (1 - corr)


def evaluate(model, loader, device, MOS_MEAN, MOS_STD, return_preds=False):
    model.eval()
    ys, y_hats = [], []
    with torch.no_grad():
        for wav, mask, y in loader:
            wav, mask = wav.to(device), mask.to(device)
            z_hat = model(wav, mask)
            mos_hat = (z_hat * MOS_STD + MOS_MEAN).clamp(1.0, 5.0)

            ys.append(y.numpy())
            y_hats.append(mos_hat.cpu().numpy())

    ys = np.concatenate(ys)
    y_hats = np.concatenate(y_hats)

    mae = np.mean(np.abs(ys - y_hats))
    rmse = np.sqrt(np.mean((ys - y_hats) ** 2))
    pearson = pearsonr(ys, y_hats)[0]
    spearman = spearmanr(ys, y_hats)[0]

    if return_preds:
        return mae, rmse, pearson, spearman, ys, y_hats
    return mae, rmse, pearson, spearman


# ============================
# Checkpointing
# ============================

def save_checkpoint(path, epoch, model, optimizer, best_pearson):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_pearson": best_pearson
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, device):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print("Warning: optimizer state not loaded (likely param groups changed).", e)

    best_pearson = ckpt.get("best_pearson", None)
    if hasattr(best_pearson, "item"):
        best_pearson = best_pearson.item()

    return ckpt["epoch"] + 1, best_pearson


# ============================
# Training
# ============================

def train(model, train_loader, val_loader, device, MOS_MEAN, MOS_STD, args):
    model.to(device)

    optimizer = torch.optim.Adam(
        list(model.regressor.parameters()) + list(model.norm.parameters()),
        lr=args.lr
    )

    start_epoch = 0
    best_pearson = -1

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_pearson = load_checkpoint(args.resume, model, optimizer, device)
        print(f"Resumed from epoch {start_epoch} | best pearson so far {best_pearson}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for wav, mask, y in pbar:
            wav = wav.to(device)
            mask = mask.to(device)
            y = y.to(device)

            z_hat = model(wav, mask)
            loss = hybrid_loss(z_hat, y, MOS_MEAN, MOS_STD, alpha=0.6)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate
        val_mae, val_rmse, val_p, val_s, ys, y_hats = evaluate(
            model, val_loader, device, MOS_MEAN, MOS_STD, return_preds=True
        )

        print(
            f"Epoch {epoch} | "
            f"Loss {np.mean(losses):.4f} | "
            f"Val MAE {val_mae:.3f} | "
            f"RMSE {val_rmse:.3f} | "
            f"P {val_p:.3f} | "
            f"S {val_s:.3f} | "
            f"y_std {ys.std():.3f} | "
            f"pred_std {y_hats.std():.3f}"
        )

        # Save checkpoints
        save_checkpoint(
            f"{args.ckpt_dir}/latest.pt",
            epoch, model, optimizer, best_pearson
        )

        if val_p > best_pearson:
            best_pearson = val_p
            save_checkpoint(
                f"{args.ckpt_dir}/best.pt",
                epoch, model, optimizer, best_pearson
            )
            print(f"New best Pearson: {best_pearson:.4f}")

    print(f"Training complete. Best Pearson: {best_pearson:.4f}")


# ============================
# Data Download
# ============================

def download_dataset(data_root):
    """Download SingMOS dataset from HuggingFace"""
    if os.path.exists(data_root) and len(os.listdir(data_root)) > 0:
        print(f"Dataset directory {data_root} already exists. Skipping download.")
        return

    print("Downloading SingMOS dataset from HuggingFace...")
    os.makedirs(data_root, exist_ok=True)

    # Use huggingface_hub to download
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="TangRain/SingMOS",
            local_dir=data_root,
            repo_type="dataset"
        )
        print(f"Dataset downloaded to {data_root}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually clone the dataset:")
        print(f"  git clone https://huggingface.co/datasets/TangRain/SingMOS {data_root}")
        sys.exit(1)


# ============================
# Main
# ============================

def main():
    args = get_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Download dataset if needed
    if args.download_data:
        download_dataset(args.data_root)

    # Check if dataset exists
    split_file = os.path.join(args.data_root, "info", "split.json")
    score_file = os.path.join(args.data_root, "info", "score.json")

    if not os.path.exists(split_file) or not os.path.exists(score_file):
        print(f"Error: Dataset not found at {args.data_root}")
        print("Run with --download_data to download the dataset")
        sys.exit(1)

    # Load data splits
    print("Loading data splits...")
    with open(split_file) as f:
        split_data = json.load(f)["singmos"]

    with open(score_file) as f:
        score_data = json.load(f)["utterance"]

    # Build item lists
    train_items = build_items(split_data["train"], score_data, args.data_root)
    val_items = build_items(split_data["valid"], score_data, args.data_root)

    print(f"Train items: {len(train_items)}")
    print(f"Val items: {len(val_items)}")

    # Compute MOS statistics
    train_mos_arr = np.array([mos for _, mos in train_items], dtype=np.float32)
    MOS_MEAN = float(train_mos_arr.mean())
    MOS_STD = float(train_mos_arr.std() + 1e-6)
    print(f"MOS mean: {MOS_MEAN:.4f}, std: {MOS_STD:.4f}")

    # Create datasets
    train_ds = SingMOSDataset(train_items)
    val_ds = SingMOSDataset(val_items)

    # Create sampler with bin weights
    train_mos = [mos for _, mos in train_items]
    weights, counts = make_bin_weights(train_mos)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    print(f"Train bin counts [1-2,2-3,3-4,4-5]: {counts.astype(int).tolist()}")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    # Build model
    print("Building model...")
    encoder = MERTEncoder(model_name=args.model_name, device=device)
    model = SingMOSModel(encoder).to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    train(model, train_loader, val_loader, device, MOS_MEAN, MOS_STD, args)

    print("\nTraining complete!")
    print(f"Best checkpoint: {args.ckpt_dir}/best.pt")
    print(f"Latest checkpoint: {args.ckpt_dir}/latest.pt")


if __name__ == "__main__":
    main()
