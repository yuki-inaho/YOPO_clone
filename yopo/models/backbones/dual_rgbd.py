import torch
import torch.nn as nn
from mmengine.model import BaseModule

from yopo.registry import MODELS


@MODELS.register_module()
class RGBDDualBackbone(BaseModule):
    """Dual-path backbone for RGB-D data.

    Runs an RGB image (3 channels) through one backbone and the depth map
    (1 channel) through a second, lighter backbone, then fuses the two
    modalities at every output scale via ``Projection -> Concat -> Conv1x1``
    (channel-wise stacking). The fused multi-scale features are returned in a
    single list so that the existing neck (e.g. ChannelMapper) and the shared
    DeformableDETR encoder can be reused unchanged.

    The input is expected to be an (N, 4, H, W) tensor (RGB + depth) as
    produced by the existing RGB-D pipeline (``ConcatDepthToImage``): the
    module splits ``x[:, :3]`` (RGB) and ``x[:, 3:4]`` (depth) internally, so
    no pipeline change is required.

    Args:
        rgb_backbone (dict): Config for the RGB backbone (e.g. HGNetV2 with
            ``in_channels=3``).
        depth_backbone (dict): Config for the depth backbone (e.g. HGNetV2
            with ``in_channels=1``). Both backbones must use the same
            ``return_idx`` so their scale lists align.
        out_channels (int): Number of output channels per fused scale.
            Defaults to 256.
        norm_cfg (dict, optional): Config for an ``nn.GroupNorm``-style
            normalization applied after each fusion ``Conv1x1``. Defaults to
            None (no normalization).
        init_cfg (dict, optional): Initialization config for this module.
    """

    def __init__(self,
                 rgb_backbone,
                 depth_backbone,
                 out_channels=256,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels

        self.rgb_backbone = MODELS.build(rgb_backbone)
        self.depth_backbone = MODELS.build(depth_backbone)

        rgb_return_idx = sorted(self.rgb_backbone.return_idx)
        depth_return_idx = sorted(self.depth_backbone.return_idx)
        assert rgb_return_idx == depth_return_idx, (
            'RGB and depth backbones must use the same return_idx, got '
            f'{rgb_return_idx} vs {depth_return_idx}')

        rgb_feats = [
            self.rgb_backbone._out_channels[i] for i in rgb_return_idx
        ]
        depth_feats = [
            self.depth_backbone._out_channels[i] for i in depth_return_idx
        ]

        self.num_scales = len(rgb_feats)
        norm = None
        if norm_cfg is not None:
            groups = norm_cfg.get('num_groups', 32)
            norm = lambda nc: nn.GroupNorm(groups, nc)

        self.rgb_proj = nn.ModuleList()
        self.depth_proj = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.fuse_norm = nn.ModuleList()
        for cin_r, cin_d in zip(rgb_feats, depth_feats):
            self.rgb_proj.append(
                nn.Conv2d(cin_r, out_channels, kernel_size=1))
            self.depth_proj.append(
                nn.Conv2d(cin_d, out_channels, kernel_size=1))
            self.fuse.append(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1))
            self.fuse_norm.append(
                nn.Identity() if norm is None else norm(out_channels))

    def forward(self, x):
        """Forward.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 4, H, W).

        Returns:
            list[torch.Tensor]: Fused feature maps, one per scale.
        """
        assert x.dim() == 4, f'expected (N, 4, H, W), got {tuple(x.shape)}'
        assert x.shape[1] == 4, (
            'RGBDDualBackbone expects a 4-channel (RGB+depth) input, '
            f'got {x.shape[1]} channels')
        x_rgb = x[:, :3, :, :]
        x_depth = x[:, 3:4, :, :]

        rgb_feats = self.rgb_backbone(x_rgb)
        depth_feats = self.depth_backbone(x_depth)

        assert len(rgb_feats) == self.num_scales
        assert len(depth_feats) == self.num_scales

        outs = []
        for i in range(self.num_scales):
            c = self.rgb_proj[i](rgb_feats[i])
            d = self.depth_proj[i](depth_feats[i])
            f = torch.cat([c, d], dim=1)
            f = self.fuse[i](f)
            f = self.fuse_norm[i](f)
            outs.append(f)
        return outs
