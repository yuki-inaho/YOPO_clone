_base_ = ['./nocs_custom_real_hgnetv2_rgbd_deim.py']

# ─────────────────────────────────────────────────────────────────────────────
# Dual-path RGB-D config.
#
# Splits the 4-channel (RGB+depth) input into two modality backbones:
#   RGB  (3ch)  -> HGNetV2-B2 (pretrained 3ch)          -> C_i [384,768,1536]
#   depth(1ch)  -> HGNetV2-B0 (pretrained, in_channels=1, non-strict)
#                                                       -> D_i [256,512,1024]
# Both use return_idx=[1,2,3] (3 scales, stride 8/16/32). At every scale the
# features are projected to 256ch and fused by channel stacking
# (Concat -> Conv1x1 -> 256ch). The merged 3-level features then go through
# the unchanged ChannelMapper (num_outs=4) into the shared DeformableDETR
# encoder. encoder/decoder/head are unchanged.
# ─────────────────────────────────────────────────────────────────────────────

model = dict(
    backbone=dict(
        _delete_=True,
        type='RGBDDualBackbone',
        out_channels=256,
        rgb_backbone=dict(
            type='HGNetV2',
            name='B2',
            in_channels=3,
            return_idx=[1, 2, 3],
            freeze_at=0,
            freeze_norm=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://github.com/Peterande/storage/releases/'
                'download/dfinev1.0/PPHGNetV2_B2_stage1.pth')),
        depth_backbone=dict(
            type='HGNetV2',
            name='B0',
            in_channels=1,
            return_idx=[1, 2, 3],
            freeze_at=0,
            freeze_norm=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://github.com/Peterande/storage/releases/'
                'download/dfinev1.0/PPHGNetV2_B0_stage1.pth'))),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 256, 256],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(use_cop_chain=True))

# Fully train from scratch on the 4-channel RGB-D dual-path backbone.
load_from = None
resume = False

# Backbone submodules are named rgb_backbone/depth_backbone in RGBDDualBackbone,
# so the paramwise custom_keys from the inherited deim config (which matched
# ``backbone``) no longer apply; re-register the slow-lr keys here.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            rgb_backbone=dict(lr_mult=0.1),
            depth_backbone=dict(lr_mult=0.1),
            encoder=dict(lr_mult=0.5))))

# The DEIM training stack (ScheduleFree, effective batch 24, cosine 100 epochs,
# TopK checkpoint, TensorBoard) is inherited from _base_.
