from types import SimpleNamespace

import torch
from torch import nn

import yopo.models.detectors.sixd_pose.dino_9d_center2d_pose as detector_module
from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
)
from yopo.models.detectors.sixd_pose.deformable_pose_detr import (
    DeformablePoseDETR,
)
from yopo.models.detectors.sixd_pose.dino_9d_center2d_pose import (
    DINO9DCenter2DPose,
)


class _ClassificationBranch(nn.Module):
    out_features = 2

    def forward(self, inputs):
        scores = inputs.new_zeros(*inputs.shape[:-1], self.out_features)
        scores[..., 0] = torch.arange(
            inputs.shape[1], device=inputs.device, dtype=inputs.dtype
        )
        return scores


class _ZeroBranch(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, inputs):
        return inputs.new_zeros(*inputs.shape[:-1], self.out_features)


class _Decoder(nn.Module):
    num_layers = 0

    def forward(self, query, reference_points, **kwargs):
        return query.unsqueeze(0), (reference_points,)


def _stub_detector_init(self, *args, **kwargs):
    nn.Module.__init__(self)
    self.as_two_stage = True
    self.with_box_refine = True
    self.embed_dims = 4
    self.num_queries = 2
    self.bbox_head = SimpleNamespace(num_classes=2)


def test_no_dn_training_uses_only_matching_queries(monkeypatch):
    monkeypatch.setattr(DeformablePoseDETR, "__init__", _stub_detector_init)
    model = DINO9DCenter2DPose(dn_cfg=None)

    assert model.dn_query_generator is None
    assert not any(key.startswith("dn_query_generator.") for key in model.state_dict())

    model.query_embedding = nn.Embedding(model.num_queries, model.embed_dims)
    model.decoder = _Decoder()
    model.bbox_head = SimpleNamespace(
        num_classes=2,
        cls_branches=[_ClassificationBranch()],
        reg_branches=[_ZeroBranch(4)],
        reg_centers_2d_branch=[_ZeroBranch(2)],
        use_bbox_for_centers_2d=False,
        cop_encoder_pose_supervision=False,
    )
    model.gen_encoder_output_proposals = lambda memory, *_: (
        memory,
        memory.new_zeros(*memory.shape[:-1], 4),
    )
    model.train()

    memory = torch.randn(2, 3, model.embed_dims)
    memory_mask = torch.zeros(2, 3, dtype=torch.bool)
    spatial_shapes = torch.tensor([[1, 3]])
    decoder_inputs, head_inputs = model.pre_decoder(
        memory, memory_mask, spatial_shapes, batch_data_samples=[]
    )

    expected_query = model.query_embedding.weight.unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(decoder_inputs["query"], expected_query)
    assert decoder_inputs["query"].shape[1] == model.num_queries
    assert decoder_inputs["reference_points"].shape[1] == model.num_queries
    assert decoder_inputs["dn_mask"] is None
    assert head_inputs["dn_meta"] is None

    decoder_outputs = model.forward_decoder(
        **decoder_inputs,
        memory_mask=memory_mask,
        spatial_shapes=spatial_shapes,
        level_start_index=torch.tensor([0]),
        valid_ratios=torch.ones(2, 1, 2),
    )
    assert decoder_outputs["hidden_states"].shape[:3] == (1, 2, 2)


def test_configured_dn_generator_keeps_existing_construction(monkeypatch):
    received = {}

    class _DnGenerator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            received.update(kwargs)
            self.label_embedding = nn.Embedding(kwargs["num_classes"], 4)

    monkeypatch.setattr(DeformablePoseDETR, "__init__", _stub_detector_init)
    monkeypatch.setattr(detector_module, "CdnQueryGenerator", _DnGenerator)
    model = DINO9DCenter2DPose(dn_cfg={"label_noise_scale": 0.5})

    assert isinstance(model.dn_query_generator, _DnGenerator)
    assert received == {
        "label_noise_scale": 0.5,
        "num_classes": 2,
        "embed_dims": 4,
        "num_matching_queries": 2,
    }
    assert "dn_query_generator.label_embedding.weight" in model.state_dict()


def test_depth_feature_extraction_returns_raw_pyramid_without_second_forward(
    monkeypatch,
):
    monkeypatch.setattr(DeformablePoseDETR, "__init__", _stub_detector_init)
    model = DINO9DCenter2DPose(dn_cfg=None)

    class _DepthBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward_with_depth_features(self, inputs):
            self.calls += 1
            return (inputs[:, :1] + 1.0,), (inputs[:, 3:4] + 2.0,)

    class _Neck(nn.Module):
        def forward(self, features):
            return tuple(feature * 3.0 for feature in features)

    model.backbone = _DepthBackbone()
    model.neck = _Neck()
    model.bbox_head = SimpleNamespace(
        num_classes=2,
        sensor_depth_scale=None,
        requires_depth_features=True,
    )
    inputs = torch.zeros(1, 4, 3, 4)

    detector_features, raw_features = model._extract_feat_with_backbone(inputs)

    assert model.backbone.calls == 1
    torch.testing.assert_close(raw_features[0], torch.ones(1, 1, 3, 4))
    torch.testing.assert_close(
        detector_features["fused_features"][0], torch.full((1, 1, 3, 4), 3.0)
    )
    torch.testing.assert_close(
        detector_features["depth_features"][0], torch.full((1, 1, 3, 4), 2.0)
    )


def test_split_outputs_accepts_missing_dn_metadata():
    predictions = (
        torch.randn(2, 1, 3, 2),
        torch.randn(2, 1, 3, 4),
        torch.randn(2, 1, 3, 2),
        torch.randn(2, 1, 3, 1),
        torch.randn(2, 1, 3, 6),
        torch.randn(2, 1, 3, 3),
    )

    no_dn_outputs = DINO9DCenter2DPoseHead.split_outputs(*predictions, dn_meta=None)
    assert all(
        actual is expected for actual, expected in zip(no_dn_outputs[:6], predictions)
    )
    assert no_dn_outputs[6:] == (None,) * 6

    dn_outputs = DINO9DCenter2DPoseHead.split_outputs(
        *predictions, dn_meta={"num_denoising_queries": 1}
    )
    assert all(output.shape[2] == 2 for output in dn_outputs[:6])
    assert all(output.shape[2] == 1 for output in dn_outputs[6:])
