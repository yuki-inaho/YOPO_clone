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

            def norm(nc):
                return nn.GroupNorm(groups, nc)

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

    def _forward_modalities(self, x):
        """Return fused maps and the projected pre-fusion depth maps."""
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
        projected_depth = []
        for i in range(self.num_scales):
            c = self.rgb_proj[i](rgb_feats[i])
            d = self.depth_proj[i](depth_feats[i])
            projected_depth.append(d)
            f = torch.cat([c, d], dim=1)
            f = self.fuse[i](f)
            f = self.fuse_norm[i](f)
            outs.append(f)
        return outs, projected_depth

    def forward(self, x):
        """Return fused feature maps, preserving the historical API."""
        fused, _ = self._forward_modalities(x)
        return fused

    def forward_with_depth_features(self, x):
        """Return fused maps and query-source depth maps explicitly.

        The depth maps are projected to ``out_channels`` but are captured
        before RGB/depth concatenation. This keeps their modality identity for
        the CoP depth-query sampler without duplicating either backbone.
        """
        return self._forward_modalities(x)


@MODELS.register_module()
class RGBDResidualBackbone(BaseModule):
    """Keep RGB features neck-compatible and add depth as a residual.

    The RGB branch is returned without projection or channel mixing.  With
    ``beta_init=0`` the module is therefore exactly equivalent to the RGB
    backbone, while the learnable depth branch can enter gradually.
    """

    def __init__(self,
                 rgb_backbone,
                 depth_backbone,
                 beta_init=0.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.rgb_backbone = MODELS.build(rgb_backbone)
        self.depth_backbone = MODELS.build(depth_backbone)

        rgb_return_idx = sorted(self.rgb_backbone.return_idx)
        depth_return_idx = sorted(self.depth_backbone.return_idx)
        if rgb_return_idx != depth_return_idx:
            raise ValueError(
                'RGB and depth backbones must use the same return_idx, got '
                f'{rgb_return_idx} vs {depth_return_idx}')

        rgb_channels = [
            self.rgb_backbone._out_channels[i] for i in rgb_return_idx
        ]
        depth_channels = [
            self.depth_backbone._out_channels[i] for i in depth_return_idx
        ]
        self.num_scales = len(rgb_channels)
        self.depth_adapters = nn.ModuleList([
            nn.Conv2d(cin_depth, cin_rgb, kernel_size=1, bias=False)
            for cin_rgb, cin_depth in zip(rgb_channels, depth_channels)
        ])
        self.depth_beta = nn.Parameter(
            torch.full((self.num_scales,), float(beta_init)))

    def _forward_modalities(self, x):
        if x.dim() != 4 or x.shape[1] != 4:
            raise ValueError(
                'RGBDResidualBackbone expects NCHW RGB+depth with 4 channels, '
                f'got {tuple(x.shape)}')
        rgb_features = tuple(self.rgb_backbone(x[:, :3]))
        depth_features = tuple(self.depth_backbone(x[:, 3:4]))
        if (len(rgb_features) != self.num_scales
                or len(depth_features) != self.num_scales):
            raise RuntimeError(
                'backbone output level count changed after construction')

        outputs = []
        for level, (rgb, depth, adapter) in enumerate(
                zip(rgb_features, depth_features, self.depth_adapters)):
            residual = adapter(depth)
            if residual.shape != rgb.shape:
                raise RuntimeError(
                    f'RGB/depth feature shape mismatch at level {level}: '
                    f'{tuple(rgb.shape)} vs {tuple(residual.shape)}')
            beta = self.depth_beta[level].to(dtype=rgb.dtype)
            outputs.append(rgb + beta * residual)
        return tuple(outputs), depth_features

    def forward(self, x):
        """Return RGB-compatible features with learned depth residuals."""
        outputs, _ = self._forward_modalities(x)
        return outputs

    def forward_with_depth_features(self, x):
        """Return fused maps and raw modality-specific depth pyramid.

        CoP consumers can project the raw depth channels independently while
        the detector neck continues to receive the pretrained-compatible RGB
        channel widths.
        """
        return self._forward_modalities(x)
