"""Ablation A4: the reduced dual-plane model, two orthogonal 2D GauCho factors.

Table 4 of the design note pairs A4 (rc pinned to 0) against A5 (rc learned) to
measure what the sixth degree of freedom is worth.  Pinning rc imposes
x2 _|_ x3 | x1 -- conditional independence of the two non-shared axes -- so the
model cannot represent a general triaxial ellipsoid.

Run this against nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageB_a5_rc.py from
the same Stage A checkpoint, same seed, same epoch budget.
"""

_base_ = ["./nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageB.py"]

model = dict(
    bbox_head=dict(
        gaucho_chart="dual_plane",
        gaucho_dual_plane_fix_rc_zero=True,
    ),
)
