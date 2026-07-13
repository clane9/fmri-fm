from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from platformdirs import user_cache_dir

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model

import cortex_mae.nisc as nisc
import cortex_mae.models_mae as models_mae


class MaeStWrapper(nn.Module):
    __space__: str = "flat"

    def __init__(self, encoder: models_mae.MaskedViT):
        super().__init__()
        self.encoder = encoder
        self.num_frames = 16
        self.max_windows = 8

    def forward(self, batch: dict[str, torch.Tensor]) -> Embeddings:
        bold = batch["bold"]
        B, C, T, H, W = bold.shape

        # pad if too short
        if T < self.num_frames:
            mean = bold.mean(dim=2, keepdim=True).expand(-1, -1, self.num_frames - T, -1, -1)
            bold = torch.cat([bold, mean], dim=2)
            T = self.num_frames

        # crop to a fixed number of non-overlapping windows
        num_windows = min(T // self.num_frames, self.max_windows)
        T = num_windows * self.num_frames
        bold = bold[:, :, :T]

        # rearrange into a batch of clips and apply model as sliding window.
        if num_windows > 1:
            bold = rearrange(bold, "b c (n f) h w -> (b n) c f h w", n=num_windows)

        cls_embeds, _, patch_embeds = self.encoder.forward_embedding(bold)

        cls_embeds = rearrange(cls_embeds, "(b n) l d -> b (n l) d", n=num_windows)
        cls_embeds = cls_embeds.mean(dim=1, keepdim=True)
        patch_embeds = rearrange(patch_embeds, "(b n) l d -> b (n l) d", n=num_windows)

        return Embeddings(cls_embeds, None, patch_embeds)


class MaeStTransform:
    def __init__(self):
        self.norm = "frame"
        self.clip_vmax = 3.0
        self.target_tr = 1.0
        self.img_size = 224

        resampler = nisc.flat_resampler_fslr64k_224_560()
        self.mask = resampler.mask_

        self.mean = torch.tensor([0.45, 0.45, 0.45])
        self.std = torch.tensor([0.225, 0.225, 0.225])

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bold = sample["bold"]
        tr = float(sample["tr"])

        bold = torch.as_tensor(bold, dtype=torch.float32)

        # temporal resample
        # nb, pretraining data used pchip interpolation, but that's very slow.
        if abs(tr - self.target_tr) > 0.1:
            bold = resample_to_tr(bold, tr=tr, target_tr=self.target_tr, mode="linear")

        # sample-wise normalization
        if self.norm:
            dim = {"frame": 1, "global": None}[self.norm]
            bold = normalize(bold, dim=dim)

        # unmask masked input
        T, D = bold.shape
        bold_ = torch.zeros(T, *self.mask.shape)
        bold_[:, self.mask] = bold
        bold = bold_  # T H W

        # gray to rgb
        bold = torch.clamp((bold + self.clip_vmax) / (2 * self.clip_vmax), 0.0, 1.0)
        bold = bold[:, None, :, :]
        bold = TF.grayscale_to_rgb(bold)  # T C H W

        # rgb normalize
        bold = (bold - self.mean[:, None, None]) / self.std[:, None, None]

        # resize
        bold = TF.resize(bold, (self.img_size, self.img_size))

        bold = bold.transpose(0, 1)  # C T H W
        sample["bold"] = bold
        return sample


def normalize(x: torch.Tensor, dim: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + eps)
    return x


def resample_to_tr(
    x: torch.Tensor, tr: float, target_tr: float, mode: str = "linear"
) -> torch.Tensor:
    T, D = x.shape
    x = x.t().unsqueeze(0)  # [1, D, T]
    x = F.interpolate(x, size=round(tr * T / target_tr), mode=mode)
    x = x.squeeze(0).t()
    return x


def fetch_mae_st_checkpoint() -> Path:
    cache_dir = Path(user_cache_dir("brainmarks"))
    cached_file = cache_dir / "mae_st" / "mae_pretrain_vit_large_k400.pth"
    if not cached_file.exists():
        cached_file.parent.mkdir(exist_ok=True, parents=True)
        urlretrieve(
            "https://dl.fbaipublicfiles.com/video-mae/pretrain/mae_pretrain_vit_large_k400.pth",
            cached_file,
        )
    return cached_file


def convert_mae_st_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ("cls_token", "cls_token"),
        ("pos_embed_spatial", "pos_embed.weight_spatial"),
        ("pos_embed_temporal", "pos_embed.weight_temporal"),
        ("pos_embed_class", "cls_token_pos"),
        ("patch_embed.proj", "patch_embed"),
    ]

    drops = [
        "decoder_pos_embed",
        "mask_token",
        "pred_head",
        "decoder_embed",
    ]

    for name, p in state_dict.items():
        if any(name.startswith(old) for old in drops):
            continue

        for old, new in swaps:
            if name.startswith(old):
                name = name.replace(old, new)
                break

        if name == "patch_embed.weight":
            out_dict[name] = p.flatten(1)
        elif name == "pos_embed.weight_temporal":
            out_dict[name] = p.transpose(0, 1)
        else:
            out_dict[name] = p
    return out_dict


@register_model
def mae_st_vit_large():
    # verified that our ViT impl produces identical output to the original MAE-st impl.
    encoder = models_mae.MaskedViT(
        img_size=224,
        in_chans=3,
        patch_size=16,
        num_frames=16,
        t_patch_size=2,
        depth=24,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4,
        class_token=True,
        pos_embed="sep",
    )

    ckpt_path = fetch_mae_st_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_state = ckpt["model_state"]
    model_state = convert_mae_st_checkpoint(model_state)
    encoder.load_state_dict(model_state)

    model = MaeStWrapper(encoder)
    transform = MaeStTransform()
    return transform, model
