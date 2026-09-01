"""Stage A: amodal 2D GauCho ellipse on the Jun30-2025 RGB-D conversion.

The curriculum of the RGB-D GauCho-3D design note starts by establishing the
2D/visible task before any metric-geometry objective is switched on.  Here the
head regresses a scale--shape Cholesky factor per query instead of an angle,
and is supervised by the inverse-free Gaussian KLD against the maximum-area
inscribed ellipse of the annotated OBB -- byte-for-byte the same target the
2D reference implementation uses, so the resulting rotated mAP50 is directly
comparable with it.

The 3D ellipsoid branch stays off in this stage.
"""

_base_ = ["./nocs_fruits_detection_Jun30_2025_rgbd_3dbbox.py"]

# The inherited chain pairs ``num_queries=100`` with ``max_per_img=300``, a
# combination that only fails at inference time (``topk`` past the end of the
# score tensor).  This dataset also carries up to ~148 annotated stems in a
# single frame, so 100 queries cannot represent it.  256 is the value the
# Jun30 curriculum base uses for exactly this data.
max_objects = 256

model = dict(
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    bbox_head=dict(
        test_cfg=dict(max_per_img=max_objects),
        gaucho_ellipse2d=True,
        gaucho_classwise=True,
        loss_ellipse2d=dict(
            type="Ellipse2DKLDLoss",
            loss_weight=2.0,
            tau=1.0,
            include_center=True,
            # A frame whose depth is missing or whose OBB is degenerate is
            # dropped through the validity mask; the run must not stop.
            fail_on_invalid=False,
        ),
    ),
)

load_from = None
resume = False
