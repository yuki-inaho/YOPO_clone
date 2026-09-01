"""Ablation A5: dual-plane with the conditional correlation learned.

The complement of A4.  With rc = tanh(xi) free, the chart is a bijection onto
SPD(3) and recovers the degree of freedom A4 gives up.
"""

_base_ = ["./nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageB.py"]

model = dict(
    bbox_head=dict(
        gaucho_chart="dual_plane",
        gaucho_dual_plane_fix_rc_zero=False,
    ),
)
