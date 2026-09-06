"""YOLO26 plus the trained DEIM hybrid encoder as one modality branch."""

from torch import nn
from mmengine.model import BaseModule

from yopo.models.backbones.yolo26 import YOLO26Backbone
from yopo.models.necks.portable_hybrid_encoder import PortableHybridEncoderNeck
from yopo.registry import MODELS


@MODELS.register_module()
class YOLO26FeatureBackbone(BaseModule):
    """Expose DEIM's three encoded feature levels to RGB-D fusion."""

    def __init__(self, scale, in_channels=3, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.scale = scale
        self.backbone = YOLO26Backbone(scale=scale, in_channels=in_channels)
        self.encoder = PortableHybridEncoderNeck(
            in_channels=tuple(self.backbone._out_channels.values()), num_outs=3
        )
        self.return_idx = self.backbone.return_idx
        self._out_channels = {index: 256 for index in self.return_idx}

    def forward(self, inputs):
        return self.encoder(self.backbone(inputs))


@MODELS.register_module()
class EncodedPyramidNeck(BaseModule):
    """Adapt fused encoded features to YOPO and derive its fourth level."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.projections = nn.ModuleList(
            nn.Conv2d(256, 256, 1, bias=False) for _ in range(3)
        )
        for projection in self.projections:
            nn.init.dirac_(projection.weight)
        self.derived_p6 = nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)

    def forward(self, inputs):
        if len(inputs) != 3 or any(
            value.ndim != 4 or value.shape[1] != 256 for value in inputs
        ):
            raise ValueError("encoded pyramid requires three NCHW 256-channel levels")
        shared = tuple(layer(value) for layer, value in zip(self.projections, inputs))
        return (*shared, self.derived_p6(shared[-1]))
