import torch
import hydra
from omegaconf import DictConfig
from typing import List, Dict, Optional

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder


class FusedObservationEncoder(BaseObservationEncoder):
    def __init__(
        self,
        shape_meta: Dict,
        vision_encoder: Optional[DictConfig]=None,
        text_encoder: Optional[DictConfig]=None,
        state_encoder: Optional[DictConfig]=None,
    ):
        super().__init__()

        # parse shape_meta
        rgb_ports = list()
        text_ports = list()
        state_ports = list()
        state_dim = 0
        for port_name, attr in shape_meta['obs'].items():
            port_type = attr.get('type', 'unsupported')
            if port_type == 'rgb':
                rgb_ports.append(port_name)
            elif port_type == 'text':
                text_ports.append(port_name)
            elif port_type == 'state':
                state_ports.append(port_name)
                port_shape = attr['shape']
                assert len(port_shape) == 1
                state_dim += port_shape[0]
            else:
                raise ValueError(f"Unsupported port type {port_type} for port {port_name}.")

        # instantiate encoders
        if rgb_ports:
            assert vision_encoder is not None, "rgb ports found but no vision_encoder provided."
            vision_encoder = hydra.utils.instantiate(vision_encoder, shape_meta=shape_meta)
        else:
            del vision_encoder
            vision_encoder = None
        if text_ports:
            assert text_encoder is not None, "text ports found but no text_encoder provided."
            text_encoder = hydra.utils.instantiate(text_encoder, shape_meta=shape_meta)
        else:
            del text_encoder
            text_encoder = None
        if state_ports:
            assert state_encoder is not None, "state ports found but no state_encoder provided."
            state_encoder = hydra.utils.instantiate(state_encoder, shape_meta=shape_meta)
        else:
            del state_encoder
            state_encoder = None
    
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.state_encoder = state_encoder
        self.rgb_ports = rgb_ports
        self.text_ports = text_ports
        self.state_ports = state_ports
        self.patch_based = (
            self.vision_encoder is not None
            and hasattr(self.vision_encoder, "output_seq_len")
        )

    def modalities(self) -> List[str]:
        mods = []
        if self.vision_encoder is not None:
            mods.append('rgb')
        if self.text_encoder is not None:
            mods.append('text')
        if self.state_encoder is not None:
            mods.append('state')
        return mods
        
    def output_feature_dim(self) -> int:
        if not self.patch_based:
            dim = 0
            if self.vision_encoder is not None:
                dim += self.vision_encoder.output_feature_dim()
            if self.text_encoder is not None:
                dim += self.text_encoder.output_feature_dim()
            if self.state_encoder is not None:
                dim += self.state_encoder.output_feature_dim()
            return dim

        dim = None
        if self.vision_encoder is not None:
            dim = self.vision_encoder.output_feature_dim()
        if self.state_encoder is not None:
            state_dim = self.state_encoder.output_feature_dim()
            if dim is None:
                dim = state_dim
            else:
                assert state_dim == dim, (
                    f"Patch-based fusion requires state_dim == vision_dim, "
                    f"got state_dim={state_dim}, vision_dim={dim}."
                )
        if self.text_encoder is not None:
            text_dim = self.text_encoder.output_feature_dim()
            if dim is None:
                dim = text_dim
            else:
                assert text_dim == dim, (
                    f"Patch-based fusion requires text_dim == feature_dim, "
                    f"got text_dim={text_dim}, feature_dim={dim}."
                )
        assert dim is not None, "No modality encoder is configured."
        return dim

    def output_seq_len(self, n_obs_steps: int) -> int:
        if self.patch_based:
            assert self.vision_encoder is not None
            seq_len = self.vision_encoder.output_seq_len(n_obs_steps)
            seq_len += n_obs_steps
            return seq_len
        return n_obs_steps
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        if self.state_encoder is not None:
            self.state_encoder.set_normalizer(normalizer)
        if self.vision_encoder is not None:
            self.vision_encoder.set_normalizer(normalizer)
        if self.text_encoder is not None:
            self.text_encoder.set_normalizer(normalizer)

    def forward(self, obs_dict: Dict) -> torch.Tensor:
        """
        obs_dict: {
            <rgb_port>: [B, To, H, W, 3] uint8/float32 tensor in [0, 255]
            <text_port>:  list of B strings
            <state_port>: [B, To, Ds] float32 tensor
        }
        """
        if not self.patch_based:
            feats = []
            To = 1
            if self.vision_encoder is not None:
                vision_feat = self.vision_encoder(obs_dict) # [B, To, Dv]
                To = vision_feat.shape[1]
                feats.append(vision_feat)
            if self.state_encoder is not None:
                state_feat = self.state_encoder(obs_dict)   # [B, To, Ds]
                To = state_feat.shape[1]
                feats.append(state_feat)
            if self.text_encoder is not None:
                text_feat = self.text_encoder(obs_dict)     # [B, 1, Dt]
                text_feat = text_feat.expand(-1, To, -1)    # [B, To, Dt]
                feats.append(text_feat)
            fused_feat = torch.cat(feats, dim=-1)           # [B, To, Dv+Dt+Ds]
            return fused_feat

        # Patch-based fusion: keep feature dimension fixed, concatenate tokens on sequence axis.
        seq_feats = []
        feature_dim = None
        To = None

        if self.vision_encoder is not None:
            vision_feat = self.vision_encoder(obs_dict)     # [B, Sv, D]
            feature_dim = vision_feat.shape[-1]
            seq_feats.append(vision_feat)

        if self.state_encoder is not None:
            state_feat = self.state_encoder(obs_dict)       # [B, To, D]
            To = state_feat.shape[1]
            if feature_dim is None:
                feature_dim = state_feat.shape[-1]
            else:
                assert state_feat.shape[-1] == feature_dim, (
                    f"Patch-based fusion requires state dim == vision dim. "
                    f"Got state dim {state_feat.shape[-1]} and vision dim {feature_dim}."
                )
            seq_feats.append(state_feat)

        if self.text_encoder is not None:
            text_feat = self.text_encoder(obs_dict)         # [B, 1, Dt]
            if To is None:
                sample = next(iter(obs_dict.values()))
                To = sample.shape[1]
            text_feat = text_feat.expand(-1, To, -1)        # [B, To, Dt]
            if feature_dim is None:
                feature_dim = text_feat.shape[-1]
            else:
                assert text_feat.shape[-1] == feature_dim, (
                    f"Patch-based fusion requires text dim == feature dim. "
                    f"Got text dim {text_feat.shape[-1]} and feature dim {feature_dim}."
                )
            seq_feats.append(text_feat)

        assert seq_feats, "No features produced by any encoder."
        fused_feat = torch.cat(seq_feats, dim=1)           # [B, S_total, D]
        return fused_feat
