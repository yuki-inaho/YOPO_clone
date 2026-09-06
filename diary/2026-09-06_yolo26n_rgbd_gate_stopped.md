# YOLO26n RGB-D YOPO diagnostic gate and stop record

## Scope

The RGB branch is YOLO26n layers 0--10 only. It does not include the YOLO OBB
head or the head-side PAN/FPN layers. The depth branch, fusion boundary, and
existing 2D/3D task heads remain YOPO-native.

## Initialization and capacity

The RGB branch received the same-scale rotated n best backbone through a strict
200-leaf transfer. Train-only frontend calibration changed seven boundary leaves
and improved held-out error by 15.7%, 27.4%, and 42.6% across the three levels.

The B30 capacity probe reported a framework peak of 28,498 MiB, but the longer
200-update gate observed 30,084 MiB resident memory in the driver. This exceeds
the 29,346 MiB safety limit, so B30 is not eligible for a FULL run.

## Gate result

The 200-update gate completed and its best checkpoint was independently
re-evaluated on 181 validation images:

| Metric | Value |
| :--- | ---: |
| HBB AP50 | 0.4479477108 |
| ellipse mAP50 | 0.4453349710 |
| projection AP50 | 0.4252622382 |
| OBB mAP50 | 0.1059535593 |
| shared AP25 | 0.1210763019 |
| 3D IoU25 | 0.1009158873 |
| invalid predictions | 0 |

The fresh-model evaluation reproduced these values. The model has 27,510,260
trainable parameters: 1,365,472 in the RGB branch, 1,837,836 in the depth
branch, 360,451 in the fusion boundary, and 23,946,501 in the remaining heads.

## Decision

The user stopped additional YOPO training after the gate. This artifact is a
diagnostic checkpoint, not a production model and not a Release candidate. A
future restart requires explicit user direction, a driver-resident capacity
search below B30, train-only calibration, and a fresh gate before any FULL run.
