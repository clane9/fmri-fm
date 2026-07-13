import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from transformers import AutoVideoProcessor, AutoModel

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model

import cortex_mae.nisc as nisc
from cortex_mae.visualization import FC_CMAP


HF_REPO = "facebook/vjepa2-vitl-fpc64-256"


class Vjepa2Wrapper(nn.Module):
    __space__: str = "flat"

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(HF_REPO)
        self.num_frames = 16
        self.max_windows = 8

    def forward(self, batch: dict[str, torch.Tensor]) -> Embeddings:
        bold = batch["bold"]
        B, T, C, H, W = bold.shape

        # pad if too short
        if T < self.num_frames:
            mean = bold.mean(dim=1, keepdim=True).expand(-1, self.num_frames - T, -1, -1, -1)
            bold = torch.cat([bold, mean], dim=1)
            T = self.num_frames

        # crop to a fixed number of non-overlapping windows
        num_windows = min(T // self.num_frames, self.max_windows)
        T = num_windows * self.num_frames
        bold = bold[:, :T]

        # rearrange into a batch of clips and apply model as sliding window.
        if num_windows > 1:
            bold = rearrange(bold, "b (n f) c h w -> (b n) f c h w", n=num_windows)

        patch_embeds = self.encoder.get_vision_features(bold)
        patch_embeds = rearrange(patch_embeds, "(b n) l d -> b (n l) d", n=num_windows)

        return Embeddings(None, None, patch_embeds)


class Vjepa2Transform:
    def __init__(self, cmap_name: str = "none"):
        self.cmap_name = cmap_name

        self.norm = "frame"
        self.clip_vmax = 3.0
        self.target_tr = 1.0
        self.img_size = 256

        if cmap_name == "none":
            self.cmap = None
        elif cmap_name == "fc":
            self.cmap = FC_CMAP
        else:
            self.cmap = plt.get_cmap(cmap_name)
            self.cmap.set_bad("gray")

        resampler = nisc.flat_resampler_fslr64k_224_560()
        self.mask = resampler.mask_

        self.num_frames = 16
        self.max_windows = 8

        self.processor = AutoVideoProcessor.from_pretrained(HF_REPO)
        self.processor.size = {"shortest_edge": 256}
        self.processor.crop_size = (256, 256)
        self.processor.do_center_crop = False

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

        # clipping
        bold = torch.clamp(bold, min=-self.clip_vmax, max=self.clip_vmax)

        # unmask masked input
        T, D = bold.shape
        bold_ = torch.zeros(T, *self.mask.shape)
        bold_[:, self.mask] = bold
        bold = bold_

        # apply colormap
        bold = bold.numpy()
        bold = np.clip((bold + self.clip_vmax) / (2 * self.clip_vmax), 0.0, 1.0)
        if self.cmap is not None:
            bold = self.cmap(bold)
            bold = bold[..., :3].transpose((0, 3, 1, 2))  # T H W C -> T C H W
        else:
            bold = bold[:, None, :, :]

        # resize
        bold = torch.from_numpy(bold)
        bold = TF.resize(bold, (self.img_size, self.img_size))
        bold = (bold * 255).to(torch.uint8)

        # vjepa2 processing
        bold = self.processor(bold, return_tensors="pt")["pixel_values_videos"]
        sample["bold"] = bold.squeeze(0)
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


@register_model
def vjepa2(*, cmap_name: str = "none") -> tuple[Vjepa2Transform, Vjepa2Wrapper]:
    transform = Vjepa2Transform(cmap_name=cmap_name)
    model = Vjepa2Wrapper()
    return transform, model
