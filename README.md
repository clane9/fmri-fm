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

## Experiments

Experiments are organized under `experiments/`, each with its own set of scripts

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
