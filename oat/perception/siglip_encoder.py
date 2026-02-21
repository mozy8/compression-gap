import contextlib
from typing import Dict, List

import torch
import torch.nn.functional as F
try:
    from transformers import SiglipVisionModel
except ImportError:
    SiglipVisionModel = None

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder


class SigLIPVisionEncoder(BaseObservationEncoder):
    """
    SigLIP vision encoder for RGB observations.

    Input:
        obs_dict[rgb_key]: [B, To, H, W, C]
    Output:
        pool_mode="avg":    [B, To * N_cam, D]
        pool_mode="tokens": [B, To * N_cam * N_patch, D]
    """

    def __init__(
        self,
        shape_meta: Dict,
        model_name: str = "google/siglip-so400m-patch14-224",
        freeze: bool = True,
        pool_mode: str = "avg",
    ):
        super().__init__()
        if pool_mode not in ("avg", "tokens"):
            raise ValueError(f"Unsupported pool_mode: {pool_mode}. Use 'avg' or 'tokens'.")

        rgb_keys = []
        for key, attr in shape_meta["obs"].items():
            if attr.get("type", None) == "rgb":
                rgb_keys.append(key)
        assert rgb_keys, "No rgb port found in shape_meta."
        if SiglipVisionModel is None:
            raise ImportError(
                "SigLIPVisionEncoder requires `transformers` with SiglipVisionModel support."
            )

        self.model = SiglipVisionModel.from_pretrained(model_name)
        self.model_name = model_name
        self.freeze = freeze
        self.pool_mode = pool_mode
        self.rgb_keys = rgb_keys
        self.n_cameras = len(rgb_keys)
        self.hidden_dim = int(self.model.config.hidden_size)
        self.image_size = int(self.model.config.image_size)
        self.patch_size = int(self.model.config.patch_size)
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.normalizer = LinearNormalizer()

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def modalities(self) -> List[str]:
        return ["rgb"]

    def output_feature_dim(self) -> int:
        return self.hidden_dim

    def output_seq_len(self, n_obs_steps: int) -> int:
        if self.pool_mode == "avg":
            return n_obs_steps * self.n_cameras
        return n_obs_steps * self.n_cameras * self.n_patches

    def set_normalizer(self, normalizer: LinearNormalizer):
        # Kept for interface compatibility. SigLIP uses fixed normalization.
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _preprocess_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: [B, To, H, W, C], uint8 in [0, 255] or float in [0, 1]/[0, 255]
        returns: [B * To, 3, image_size, image_size], float32
        """
        assert rgb.dim() == 5, f"Expected [B, To, H, W, C], got shape {tuple(rgb.shape)}."
        assert rgb.shape[-1] == 3, f"Expected 3 channels, got {rgb.shape[-1]}."

        B, To, H, W, _ = rgb.shape
        x = rgb.reshape(B * To, H, W, 3).permute(0, 3, 1, 2).contiguous()

        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = x.float()
            if x.max() > 1.0:
                x = x.div(255.0)

        if (H, W) != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        # SigLIP normalization
        x = (x - 0.5) / 0.5
        return x

    def forward(self, obs_dict: Dict) -> torch.Tensor:
        sample = obs_dict[self.rgb_keys[0]]
        B, To = sample.shape[:2]

        camera_batches = []
        for key in self.rgb_keys:
            if key not in obs_dict:
                raise KeyError(f"Missing rgb observation key: {key}")
            rgb = obs_dict[key]
            assert rgb.shape[:2] == (B, To), (
                f"All rgb ports must share [B, To]. "
                f"Expected {(B, To)} for {key}, got {tuple(rgb.shape[:2])}."
            )
            camera_batches.append(self._preprocess_rgb(rgb))

        pixel_values = torch.stack(camera_batches, dim=1)
        pixel_values = pixel_values.reshape(
            B * To * self.n_cameras, 3, self.image_size, self.image_size
        )

        grad_ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with grad_ctx:
            outputs = self.model(pixel_values=pixel_values)
            patch_tokens = outputs.last_hidden_state

        n_patches = patch_tokens.shape[1]
        if n_patches != self.n_patches:
            self.n_patches = n_patches

        patch_tokens = patch_tokens.reshape(B, To, self.n_cameras, n_patches, self.hidden_dim)

        if self.pool_mode == "avg":
            feat = patch_tokens.mean(dim=3)  # [B, To, N_cam, D]
            return feat.reshape(B, To * self.n_cameras, self.hidden_dim)

        feat = patch_tokens.reshape(B, To * self.n_cameras * n_patches, self.hidden_dim)
        return feat
