# YOLO26m RGB-D transfer design

## Scope

The production graph is:

```text
RGB -> YOLO26m backbone layers 0-10 -- P3/P4/P5 --+
                                                    +-> residual RGB-D fusion
Depth -> HGNetV2-B0 -> 1x1 adapters ---------------+   -> PortableHybridEncoder
                                                        -> YOPO query decoder and 2D/3D heads
```

The native YOLO Detect/OBB head and the YOLO head-side PAN/FPN layers 11-22 are
not part of this graph. The DEIMv2 encoder and decoder weights are not copied
into YOPO. Only the accepted DEIMv2 JAX model's YOLO26m backbone initializes the
new RGB branch.

## Transfer contract

| Contract | Value |
|---|---|
| Source | accepted DEIMv2 JAX YOLO26m-backbone model |
| Source state | `ema::backbone/**` |
| Required source leaves | exactly 250 |
| YOLO layers | 0-10 only |
| Feature taps | layers 4, 6, 10 |
| Feature shapes | P3/P4/P5, channels 512/512/512, strides 8/16/32 |
| Kernel mapping | JAX HWIO to PyTorch OIHW |
| BatchNorm mapping | affine parameters and running statistics |
| PyTorch-only state | 50 `num_batches_tracked` leaves, explicitly initialized |
| Required parity | all 250 leaves mapped, no extra source leaf, three feature levels numerically close |

The measured JAX/PyTorch feature parity had mean absolute errors
`1.67e-4 / 2.11e-4 / 2.18e-4` and maximum absolute errors
`0.00532 / 0.00489 / 0.00282` for P3/P4/P5.

## YOPO initialization boundary

The initial 1,167-leaf YOPO state is composed of:

- 250 transferred YOLO26m RGB-backbone leaves;
- 860 same-name, same-shape leaves from the validated stage-8 YOPO model;
- 57 fresh or explicitly reset leaves at changed interfaces.

The stage-8 allowlist retains the depth backbone, detector encoder/decoder,
query and memory embeddings, task heads, and the shape-compatible part of the
neck. It excludes the old RGB backbone, DEIMv2 encoder/decoder, depth adapters,
`depth_beta`, and input projections whose shapes changed.

## Train-only boundary calibration

An uncalibrated full run stayed finite but failed every promotion guard because
the random changed-width boundary destroyed the inherited feature basis. The
repair does not alter the architecture, head, loss, or validation data:

1. Collect paired stage-8 teacher and YOLO26m student RGB features from the
   training split only, without labels or random transforms.
2. Fit the three `neck.projections.*.weight` tensors by ridge regression.
3. Factor the teacher depth contribution into the three
   `backbone.depth_adapters.*.weight` tensors and copy `backbone.depth_beta`.
4. Fail closed unless exactly these seven state leaves change and all values
   are finite.

The fit used 16 batches of 4 training images, 4 disjoint holdout batches,
2,048 samples per level, projection ridge `1e-4`, depth ridge `1e-6`, and seed
`20260906`. Holdout relative RMSE improved from
`1.0117 / 1.0056 / 1.0041` to `0.6386 / 0.6306 / 0.5253`.

## Training and selection contract

Training uses batch 24, BF16, AMUSE, 50 epochs maximum, validation every five
epochs, and early stopping on `projection/shared_AP_25` with `min_delta=0.001`
and patience 4. Batch 25 was finite but exceeded the fixed 29,346 MiB operating
limit; batch 24 peaked at 29,154 MiB and is the largest accepted batch.

The selected checkpoint must satisfy all conditions at one validation epoch:

- maximize `projection/shared_AP_25`;
- HBB AP50 at least `0.8514`;
- ellipse mAP50 at least `0.8506`;
- projection AP50 at least `0.8188`;
- zero invalid ellipsoid and projection predictions.

Epoch 10 is selected. It improves shared AP25 from the revalidated stage-8
baseline `0.2208` to `0.2679015474`, while passing all 2D guards. Strict 3D
IoU25 is `0.1914560553`, a `0.004544` absolute decrease from stage 8; this is
reported rather than hidden. The run stopped normally at epoch 30 after four
consecutive primary-metric non-improvements.

## Fail-closed rules

- Reject source trees with missing, extra, duplicate, non-finite, or
  shape-incompatible transferred leaves.
- Do not silently load a mismatched state or fall back to random transfer.
- Do not use validation images or labels during boundary calibration.
- Do not promote a checkpoint on a 2D metric alone.
- Release exactly one model checkpoint: the selected epoch-10 primary best.
