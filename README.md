# SingMOS Training & Inference

MOS (Mean Opinion Score) prediction model for singing voice synthesis evaluation, using MERT embeddings.

## Files

- `train_singmos.py` - Training script
- `predict_mos.py` - Inference script for predicting MOS on audio files
- `requirements.txt` - Python dependencies

## Setup on Lambda GPU

### 1. SSH into your Lambda instance

```bash
ssh ubuntu@<your-lambda-ip>
```

### 2. Check GPU availability

```bash
nvidia-smi
```

You should see your GPU (e.g., A10, A100, H100) listed.

### 3. Check if CUDA is installed

```bash
nvcc --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Create a virtual environment

```bash
cd ~
python3 -m venv singmos_env
source singmos_env/bin/activate
```

### 5. Upload/download the code

Option A: Clone from GitHub (if you push these files)
```bash
git clone <your-repo-url>
cd singmos
```

Option B: Use SCP to upload files from local machine
```bash
# From your local machine:
scp train_singmos.py predict_mos.py requirements.txt ubuntu@<lambda-ip>:~/
ssh ubuntu@<lambda-ip>
source ~/singmos_env/bin/activate
```

Option C: Use VS Code Remote-SSH extension to edit files directly on the server

### 6. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you want to use specific CUDA versions, you can install PyTorch with CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 7. Download the dataset

The dataset is hosted on HuggingFace. Run:

```bash
python train_singmos.py --download_data --epochs 0
```

This will download the SingMOS dataset to `./SingMOS/`.

Alternatively, manually clone:
```bash
git clone https://huggingface.co/datasets/TangRain/SingMOS
```

## Training

### Basic training

```bash
python train_singmos.py \
    --data_root ./SingMOS \
    --ckpt_dir ./checkpoints \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda
```

### Resume from checkpoint

```bash
python train_singmos.py \
    --data_root ./SingMOS \
    --ckpt_dir ./checkpoints \
    --epochs 50 \
    --resume ./checkpoints/latest.pt \
    --device cuda
```

### Training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `./SingMOS` | Path to dataset |
| `--ckpt_dir` | `./checkpoints` | Checkpoint save directory |
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--device` | cuda | Device (cuda or cpu) |
| `--resume` | None | Path to checkpoint to resume |
| `--download_data` | False | Download dataset from HuggingFace |
| `--model_name` | m-a-p/MERT-v1-95M | MERT model variant |

## Inference / Prediction

### Predict on a single audio file

```bash
python predict_mos.py \
    --audio /path/to/singing.wav \
    --ckpt ./checkpoints/best.pt \
    --device cuda
```

### Predict on a directory of audio files

```bash
python predict_mos.py \
    --audio /path/to/audio_folder/ \
    --ckpt ./checkpoints/best.pt \
    --device cuda
```

### If MOS stats not in checkpoint

```bash
python predict_mos.py \
    --audio /path/to/audio.wav \
    --ckpt ./checkpoints/best.pt \
    --mos_mean 3.2341 \
    --mos_std 0.8123 \
    --device cuda
```

## Example Output

Training:
```
Using device: cuda
GPU: NVIDIA A10
CUDA version: 12.1
Train items: 3200
Val items: 800
MOS mean: 3.2341, std: 0.8123
Epoch 0 | Loss 0.5234 | Val MAE 0.312 | RMSE 0.421 | P 0.823 | S 0.812 | y_std 0.823 | pred_std 0.765
New best Pearson: 0.8234
```

Prediction:
```
Using device: cuda
Loading checkpoint: ./checkpoints/best.pt
Processing: /path/to/singing.wav
Predicted MOS: 3.456
```

## Lambda GPU Specific Tips

### Persistent training with tmux/screen

To keep training running when you disconnect:

```bash
# Install tmux if not present
sudo apt-get update && sudo apt-get install -y tmux

# Create new session
tmux new -s singmos_training

# Run training
source ~/singmos_env/bin/activate
cd ~/singmos
python train_singmos.py --epochs 50 --batch_size 16

# Detach: Press Ctrl+B, then D

# Reconnect later
tmux attach -t singmos_training
```

### Monitor GPU usage

In another terminal:
```bash
watch -n 1 nvidia-smi
```

### Download checkpoints to local

```bash
scp ubuntu@<lambda-ip>:~/singmos/checkpoints/best.pt ./
```

### Common Lambda Issues

1. **Out of memory**: Reduce `--batch_size` (try 8 instead of 16)
2. **Slow data loading**: The dataset has many small files. First epoch may be slow due to HuggingFace cache downloading.
3. **Disk space**: Dataset is ~several GB. Check disk space with `df -h`

## Model Architecture

- **Encoder**: MERT-v1-95M (frozen during training)
- **Pooling**: Mean + Std pooling over time
- **Head**: LayerNorm + Linear(2D, 512) + ReLU + Dropout(0.2) + Linear(512, 1)
- **Loss**: Hybrid loss (L1 + Pearson correlation)

## Citation

If you use this code or the SingMOS dataset:

```bibtex
@dataset{tangrain_singmos,
  author = {TangRain},
  title = {SingMOS Dataset},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/datasets/TangRain/SingMOS}}
}
```

## Troubleshooting

### CUDA out of memory
```bash
python train_singmos.py --batch_size 8  # Reduce batch size
```

### Dataset not found
```bash
python train_singmos.py --download_data  # Download dataset
```

### Model download fails
The MERT model downloads from HuggingFace. If behind a firewall:
```bash
export HF_ENDPOINT=https://hf-mirror.com  # For China users
python train_singmos.py
```
