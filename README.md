# CortexMAE

Masked autoencoder (MAE) pretraining for fMRI data using cortical flat map representations.

## Structure

```
src/flat_mae/
├── main_pretrain.py          # Training entrypoint
├── models_mae.py             # MAE model implementation
├── data.py                   # Data loading
├── transforms.py             # Data augmentations
├── masking.py                # Masking strategies
└── config/
    └── default_pretrain.yaml # Default configuration
```

## Installation

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Experiments

Experiments are organized under `experiments/`, each with its own config and launch script:

```
experiments/
├── input_space/      # Input representation ablation
├── model_scaling/    # Model size scaling
├── data_scaling/     # Dataset size scaling
├── masking/          # Masking strategy ablation
├── decoders/         # Decoder architecture ablation
├── augmentation/     # Data augmentation ablation
├── t_patch_size/     # Temporal patch size ablation
├── pos_embed/        # Position embedding ablation
└── target_norm/      # Target normalization ablation
```

## Usage

Run an experiment via its launch script:

```bash
bash experiments/input_space/launch_pretrain.sh
```

Each experiment folder contains:
- `pretrain.yaml` - experiment-specific config (overrides defaults)
- `launch_pretrain.sh` - SLURM launch script
- `output/` - checkpoints and logs
