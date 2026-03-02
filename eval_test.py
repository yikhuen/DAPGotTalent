#!/usr/bin/env python3
"""
SingMOS Test Evaluation Script
Evaluate any checkpoint on the test set independently.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from train_singmos import (
    BackboneEncoder, SingMOSModel, SingMOSDataset, collate_fn,
    build_items, evaluate, set_seed
)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SingMOS model on any dataset split")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="./SingMOS", help="Path to SingMOS dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"],
                        help="Which split to evaluate on (default: test, falls back to valid if test not available)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default=None,
        choices=["wav2vec2", "mert"],
        help="Backbone encoder type; if omitted, read from checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Backbone model name; if omitted, read from checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--rms_norm", action="store_true", help="Apply RMS normalization (should match training)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Check dataset exists
    split_file = os.path.join(args.data_root, "info", "split.json")
    score_file = os.path.join(args.data_root, "info", "score.json")

    if not os.path.exists(split_file) or not os.path.exists(score_file):
        print(f"Error: Dataset not found at {args.data_root}")
        sys.exit(1)

    # Load data splits
    print("Loading data splits...")
    with open(split_file) as f:
        split_data = json.load(f)["singmos"]
    with open(score_file) as f:
        score_data = json.load(f)["utterance"]

    # Determine which split to evaluate
    eval_split = args.split
    if eval_split not in split_data:
        if eval_split == "test" and "valid" in split_data:
            print(f"Note: '{eval_split}' split not found. Falling back to 'valid' split.")
            eval_split = "valid"
        else:
            available = list(split_data.keys())
            print(f"Error: Split '{eval_split}' not found. Available splits: {available}")
            sys.exit(1)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder_type = args.encoder_type or ckpt.get("encoder_type", "wav2vec2")
    model_name = args.model_name or ckpt.get("model_name", "facebook/wav2vec2-large-960h-lv60-self")

    # Get MOS stats from checkpoint
    if "mos_mean" in ckpt and "mos_std" in ckpt:
        MOS_MEAN = ckpt["mos_mean"]
        MOS_STD = ckpt["mos_std"]
        print(f"MOS stats from checkpoint: mean={MOS_MEAN:.4f}, std={MOS_STD:.4f}")
    else:
        print("Warning: MOS stats not found in checkpoint. Using training set stats.")
        # Compute from training data as fallback
        train_items = build_items(split_data["train"], score_data, args.data_root)
        train_mos_arr = np.array([mos for _, mos in train_items], dtype=np.float32)
        MOS_MEAN = float(train_mos_arr.mean())
        MOS_STD = float(train_mos_arr.std() + 1e-6)
        print(f"Computed MOS stats: mean={MOS_MEAN:.4f}, std={MOS_STD:.4f}")

    # Load evaluation data
    eval_items = build_items(split_data[eval_split], score_data, args.data_root)
    print(f"{eval_split} items: {len(eval_items)}")

    if len(eval_items) == 0:
        print(f"Error: No items found in '{eval_split}' split")
        sys.exit(1)

    eval_ds = SingMOSDataset(
        eval_items,
        augment=False,
        normalize_rms=args.rms_norm
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    # Build and load model
    print("Building model...")
    print(f"Encoder: {encoder_type} | Model: {model_name}")
    encoder = BackboneEncoder(
        model_name=model_name,
        encoder_type=encoder_type,
        device=device
    )
    model = SingMOSModel(encoder).to(device)
    model.load_state_dict(ckpt["model"])

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Evaluate on selected split
    print(f"\nEvaluating on '{eval_split}' split...")
    mae, rmse, pearson, spearman = evaluate(model, eval_loader, device, MOS_MEAN, MOS_STD)

    # Print results
    print("\n" + "=" * 50)
    print(f"{eval_split.upper()} SET RESULTS")
    print("=" * 50)
    print(f"Checkpoint: {args.ckpt}")
    print(f"MAE:        {mae:.4f}")
    print(f"RMSE:       {rmse:.4f}")
    print(f"Pearson:    {pearson:.4f}")
    print(f"Spearman:   {spearman:.4f}")
    print("=" * 50)

    # Save results to JSON if requested
    if args.output:
        results = {
            "checkpoint": args.ckpt,
            "split": eval_split,
            "split_size": len(eval_items),
            "mos_mean": MOS_MEAN,
            "mos_std": MOS_STD,
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson": float(pearson),
            "spearman": float(spearman),
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
