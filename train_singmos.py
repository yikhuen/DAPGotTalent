#!/usr/bin/env python3
"""
SingMOS Training Script
Train a MOS (Mean Opinion Score) prediction model for singing voice synthesis evaluation.
"""

import os
import sys
import json
import random
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
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Learning rate for unfrozen encoder layers")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--model_name", type=str, default="m-a-p/MERT-v1-95M", help="MERT model name")
    parser.add_argument("--download_data", action="store_true", help="Download dataset from HuggingFace")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--unfreeze_last_n", type=int, default=2, help="Unfreeze top N transformer layers")
    parser.add_argument("--unfreeze_epoch", type=int, default=3, help="Epoch index to unfreeze top layers")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--early_stop_patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--alpha_start", type=float, default=0.9, help="Initial hybrid loss alpha")
    parser.add_argument("--alpha_end", type=float, default=0.6, help="Final hybrid loss alpha")
    parser.add_argument("--no_weighted_sampler", action="store_true", help="Disable weighted random sampling")
    parser.add_argument("--sampler_max_ratio", type=float, default=5.0, help="Max bin weight ratio for sampler")
    parser.add_argument("--train_augment", action="store_true", help="Enable light train-time waveform augmentation")
    parser.add_argument("--rms_norm", action="store_true", help="Normalize waveform RMS before model input")
    parser.add_argument("--eval_test", action="store_true", help="Evaluate test split after training (if available)")
    return parser.parse_args()

# ============================
# Dataset
# ============================

class SingMOSDataset(Dataset):
    def __init__(self, items, sample_rate=16000, augment=False, normalize_rms=False, target_rms=0.1):
        self.items = items
        self.sr = sample_rate
        self.augment = augment
        self.normalize_rms = normalize_rms
        self.target_rms = target_rms

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

        if self.normalize_rms:
            rms = torch.sqrt(torch.clamp((wav ** 2).mean(), min=1e-8))
            wav = wav * (self.target_rms / rms)

        if self.augment:
            # Light amplitude and noise jitter to improve robustness.
            gain = float(torch.empty(1).uniform_(0.85, 1.15))
            wav = wav * gain
            noise_std = float(torch.empty(1).uniform_(0.0005, 0.003))
            wav = wav + torch.randn_like(wav) * noise_std
            wav = wav.clamp(-1.0, 1.0)

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

    def _get_transformer_layers(self):
        candidates = [
            ("encoder", "layers"),
            ("wav2vec2", "encoder", "layers"),
            ("model", "encoder", "layers"),
            ("transformer", "layers"),
        ]
        for path in candidates:
            node = self.model
            ok = True
            for attr in path:
                if not hasattr(node, attr):
                    ok = False
                    break
                node = getattr(node, attr)
            if ok and hasattr(node, "__len__"):
                return node
        return None

    def unfreeze_last_n_layers(self, n_layers):
        if n_layers <= 0:
            return 0
        layers = self._get_transformer_layers()
        if layers is None:
            print("Warning: could not locate transformer layers for unfreezing.")
            return 0

        for p in self.model.parameters():
            p.requires_grad = False

        n_layers = min(n_layers, len(layers))
        for idx in range(len(layers) - n_layers, len(layers)):
            for p in layers[idx].parameters():
                p.requires_grad = True
        return n_layers

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

def save_checkpoint(path, epoch, model, optimizer, best_pearson, MOS_MEAN, MOS_STD):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_pearson": best_pearson,
        "mos_mean": MOS_MEAN,
        "mos_std": MOS_STD,
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

def build_optimizer(model, args):
    head_params = list(model.regressor.parameters()) + list(model.norm.parameters())
    param_groups = [{
        "params": head_params,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }]

    encoder_params = [p for p in model.encoder.model.parameters() if p.requires_grad]
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
        })

    return torch.optim.AdamW(param_groups)


def train(model, train_loader, val_loader, test_loader, device, MOS_MEAN, MOS_STD, args):
    model.to(device)

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.scheduler_patience
    )

    start_epoch = 0
    best_pearson = -1
    no_improve_epochs = 0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_pearson = load_checkpoint(args.resume, model, optimizer, device)
        print(f"Resumed from epoch {start_epoch} | best pearson so far {best_pearson}")

    for epoch in range(start_epoch, args.epochs):
        if args.unfreeze_last_n > 0 and epoch == args.unfreeze_epoch:
            unfrozen = model.encoder.unfreeze_last_n_layers(args.unfreeze_last_n)
            if unfrozen > 0:
                optimizer = build_optimizer(model, args)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=args.scheduler_patience
                )
                print(f"Unfroze top {unfrozen} encoder layers at epoch {epoch}")

        model.train()
        losses = []
        if args.epochs > max(start_epoch + 1, 1):
            progress = (epoch - start_epoch) / max(1, (args.epochs - start_epoch - 1))
            alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * progress
        else:
            alpha = args.alpha_end

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for wav, mask, y in pbar:
            wav = wav.to(device)
            mask = mask.to(device)
            y = y.to(device)

            z_hat = model(wav, mask)
            loss = hybrid_loss(z_hat, y, MOS_MEAN, MOS_STD, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "alpha": f"{alpha:.2f}"})

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
            f"pred_std {y_hats.std():.3f} | "
            f"alpha {alpha:.2f}"
        )
        scheduler.step(val_p)

        # Save checkpoints
        save_checkpoint(
            f"{args.ckpt_dir}/latest.pt",
            epoch, model, optimizer, best_pearson, MOS_MEAN, MOS_STD
        )

        if val_p > best_pearson:
            best_pearson = val_p
            no_improve_epochs = 0
            save_checkpoint(
                f"{args.ckpt_dir}/best.pt",
                epoch, model, optimizer, best_pearson, MOS_MEAN, MOS_STD
            )
            print(f"New best Pearson: {best_pearson:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Training complete. Best Pearson: {best_pearson:.4f}")
    if test_loader is not None:
        t_mae, t_rmse, t_p, t_s = evaluate(model, test_loader, device, MOS_MEAN, MOS_STD)
        print(
            f"Test | MAE {t_mae:.3f} | RMSE {t_rmse:.3f} | "
            f"P {t_p:.3f} | S {t_s:.3f}"
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    set_seed(args.seed)

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
    test_items = build_items(split_data["test"], score_data, args.data_root) if "test" in split_data else []

    print(f"Train items: {len(train_items)}")
    print(f"Val items: {len(val_items)}")
    if test_items:
        print(f"Test items: {len(test_items)}")

    # Compute MOS statistics
    train_mos_arr = np.array([mos for _, mos in train_items], dtype=np.float32)
    MOS_MEAN = float(train_mos_arr.mean())
    MOS_STD = float(train_mos_arr.std() + 1e-6)
    print(f"MOS mean: {MOS_MEAN:.4f}, std: {MOS_STD:.4f}")

    # Create datasets
    train_ds = SingMOSDataset(
        train_items,
        augment=args.train_augment,
        normalize_rms=args.rms_norm
    )
    val_ds = SingMOSDataset(
        val_items,
        augment=False,
        normalize_rms=args.rms_norm
    )
    test_ds = SingMOSDataset(
        test_items,
        augment=False,
        normalize_rms=args.rms_norm
    ) if test_items else None

    # Create sampler with bin weights
    train_mos = [mos for _, mos in train_items]
    weights, counts = make_bin_weights(train_mos, max_ratio=args.sampler_max_ratio)
    sampler = None if args.no_weighted_sampler else WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    print(f"Train bin counts [1-2,2-3,3-4,4-5]: {counts.astype(int).tolist()}")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=True if sampler is None else False,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    ) if test_ds is not None else None

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
    train(
        model,
        train_loader,
        val_loader,
        test_loader if args.eval_test else None,
        device,
        MOS_MEAN,
        MOS_STD,
        args
    )

    print("\nTraining complete!")
    print(f"Best checkpoint: {args.ckpt_dir}/best.pt")
    print(f"Latest checkpoint: {args.ckpt_dir}/latest.pt")


if __name__ == "__main__":
    main()
