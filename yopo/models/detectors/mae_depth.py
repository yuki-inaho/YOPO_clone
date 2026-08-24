from typing import Dict, Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import Tensor

from yopo.registry import MODELS
from yopo.structures import DetDataSample, OptSampleList, SampleList
from yopo.utils import InstanceList, OptConfigType, OptMultiConfig

from .base import BaseDetector


@MODELS.register_module()
class MAEDepth(BaseDetector):
    """Masked depth reconstruction (MAE-style) for the depth backbone.

    Self-supervised pretraining for the depth branch (HGNetV2-B0, 1-channel).
    Randomly masks pixels of the normalized depth channel, encodes the masked
    depth with the backbone, then reconstructs the original depth with a light
    decoder. The reconstruction loss is only computed on masked pixels, as in
    FCMAE / Masked Autoencoders.

    Input is the 4-channel (RGB+depth) preprocessed tensor; only the depth
    channel (index 3) is used.

    Args:
        encoder (dict): Config of the depth backbone (e.g. HGNetV2 with
            ``in_channels=1`` and ``return_idx=[1, 2, 3]``).
        hidden_channels (int): Hidden channels of the reconstruction decoder.
            Defaults to 128.
        mask_ratio (float): Fraction of depth pixels to mask. Defaults to 0.75.
        loss_weight (float): Weight of the reconstruction loss. Defaults to 1.0.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self,
                 encoder,
                 data_preprocessor: OptConfigType = None,
                 hidden_channels=128,
                 mask_ratio=0.75,
                 loss_weight=1.0,
                 init_cfg=None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.encoder = MODELS.build(encoder)
        self.hidden_channels = hidden_channels
        self.mask_ratio = mask_ratio
        self.loss_weight = loss_weight

        # Multi-scale feat channels returned by the encoder
        # (return_idx=[1,2,3] -> stage1/2/3 output channels, stride 8/16/32).
        enc_outs = list(self.encoder._out_channels)
        self._feat_channels = [enc_outs[i] for i in sorted(self.encoder.return_idx)]

        # Top-down fusion decoder from the coarsest returned scale to the
        # finest (stride 8), then a reconstruction head to input resolution.
        self.lateral = nn.ModuleList()
        self.fpn = nn.ModuleList()
        for c in reversed(self._feat_channels[:-1]):
            self.lateral.append(
                nn.Sequential(
                    nn.Conv2d(c, hidden_channels, kernel_size=1),
                    nn.GroupNorm(32, hidden_channels), nn.ReLU(inplace=True)))
            self.fpn.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels,
                              hidden_channels,
                              kernel_size=3,
                              padding=1),
                    nn.GroupNorm(32, hidden_channels), nn.ReLU(inplace=True)))
        self.top_proj = nn.Sequential(
            nn.Conv2d(self._feat_channels[-1],
                      hidden_channels,
                      kernel_size=1),
            nn.GroupNorm(32, hidden_channels), nn.ReLU(inplace=True))
        self.recon_head = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size=3,
                      padding=1),
            nn.GroupNorm(32, hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_feat(self, batch_inputs: Tensor,
                     batch_data_samples: OptSampleList = None):
        """Extract depth features from the depth channel of the input."""
        depth = batch_inputs[:, 3:4, :, :]
        return self.encoder(depth)

    def _reconstruct(self, depth: Tensor) -> Tensor:
        """Encode a depth map (1ch) and reconstruct to input resolution."""
        feats = self.encoder(depth)
        x = self.top_proj(feats[-1])
        for i in range(len(self._feat_channels) - 2, -1, -1):
            lat = self.lateral[len(self._feat_channels) - 2 - i]
            f = self.fpn[len(self._feat_channels) - 2 - i]
            target = feats[i]
            x = F.interpolate(x,
                              size=target.shape[-2:],
                              mode='bilinear',
                              align_corners=False)
            x = f(lat(target) + x)
        pred = self.recon_head(x)
        pred = F.interpolate(pred,
                             size=depth.shape[-2:],
                             mode='bilinear',
                             align_corners=False)
        return pred

    def _forward(self, batch_inputs: Tensor) -> Tensor:
        """Forward on the full input batch (uses the depth channel only)."""
        depth = batch_inputs[:, 3:4, :, :]
        return self._reconstruct(depth)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute the masked-depth reconstruction loss."""
        depth = batch_inputs[:, 3:4, :, :]
        b, _, h, w = depth.shape
        num_masked = int(self.mask_ratio * h * w)
        mask = torch.ones_like(depth)
        for i in range(b):
            idx = torch.randperm(h * w, device=depth.device)[:num_masked]
            y = idx // w
            x = idx % w
            mask[i, 0, y, x] = 0.0

        masked = depth * mask
        pred = self._reconstruct(masked)
        loss_pix = F.l1_loss(pred[mask == 0], depth[mask == 0])
        return {'loss_depth': self.loss_weight * loss_pix}

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        outs = self._forward(batch_inputs)
        for i, sample in enumerate(batch_data_samples):
            sample.set_metainfo({'pred_depth': outs[i].detach()})
        return batch_data_samples