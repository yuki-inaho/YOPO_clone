"""ONNX export/runtime primitives for the YOPO 9D pose model.

The exported graph deliberately contains the RGB preprocessing and emits the
last decoder tensors. Detection selection and camera-dependent translation
recovery stay in :func:`postprocess_pose_outputs`, which keeps the ONNX graph
portable and makes the camera intrinsics an explicit runtime input to the
deployment adapter.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch import Tensor, nn

from yopo.registry import MODELS
from yopo.structures import DetDataSample

# The graph is exported for the repository's fixed inference resolution. The
# released YOPO checkpoint was trained/evaluated with a 640x480 pipeline.
DEFAULT_INPUT_SIZE = (640, 480)  # (width, height)
ONNX_OUTPUT_NAMES = (
    'cls_logits',
    'bbox_cxcywh',
    'center_2d',
    'z',
    'rotation_6d',
    'size',
)


def build_yopo_model(config: Union[str, Config], checkpoint: str,
                     device: str = 'cpu') -> nn.Module:
    """Build a YOPO model and load a checkpoint."""
    # Importing the model package populates the project registry.
    import yopo.models  # noqa: F401

    init_default_scope('yopo')
    if isinstance(config, str):
        config = Config.fromfile(config)
    model = MODELS.build(config.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()
    return model.to(device)


class YopoOnnxWrapper(nn.Module):
    """Wrap YOPO's tensor forward with a fixed deployment input contract.

    Input is a single BGR image tensor of shape ``[1, 3, 480, 640]`` with
    float32 values in the usual 0..255 image range. The wrapper performs the
    same BGR->RGB normalization as ``DetDataPreprocessor`` and returns the
    final decoder output of each YOPO pose branch.

    The current exporter intentionally fixes batch and spatial dimensions.
    YOPO's traced proposal generation iterates over feature-map shapes, so a
    static contract is safer than claiming unsupported dynamic H/W behavior.
    """

    def __init__(self, model: nn.Module,
                 input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE) -> None:
        super().__init__()
        width, height = input_size
        if width <= 0 or height <= 0:
            raise ValueError('input_size must contain positive dimensions')
        self.model = model
        self.input_size = (int(width), int(height))
        self.register_buffer(
            'mean',
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

        sample = DetDataSample()
        sample.set_metainfo({
            'batch_input_shape': (int(height), int(width)),
            'img_shape': (int(height), int(width)),
            'ori_shape': (int(height), int(width)),
            'scale_factor': (1.0, 1.0),
        })
        self.sample = sample

    def forward(self, image: Tensor) -> Tuple[Tensor, ...]:
        if image.dim() != 4:
            raise ValueError('image must have shape [1, 3, H, W]')
        # The exported artifact is intentionally batch-1.
        if image.shape[0] != 1:
            raise ValueError('the exported YOPO graph supports batch size 1')
        image = torch.stack(
            (image[:, 2, :, :], image[:, 1, :, :], image[:, 0, :, :]),
            dim=1)
        image = (image - self.mean) / self.std
        outputs = self.model._forward(image, [self.sample])
        return tuple(output[-1] for output in outputs)


def _as_numpy(value: Union[np.ndarray, Tensor]) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def postprocess_pose_outputs(
        outputs: Sequence[Union[np.ndarray, Tensor]],
        intrinsic: Sequence[float],
        original_shape: Tuple[int, int],
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        num_classes: int = 6,
        max_per_img: int = 300,
        classwise_rotation: bool = True,
        classwise_sizes: bool = True,
        use_log_z: bool = False) -> Dict[str, np.ndarray]:
    """Convert raw ONNX/PyTorch branch outputs into YOPO predictions.

    This mirrors ``DINO9DCenter2DPoseHead._predict_by_feat_single``. The
    returned bboxes and centers are in original RGB-image coordinates and
    translations use the supplied RGB camera intrinsics.
    """
    if len(outputs) != 6:
        raise ValueError('expected six YOPO output tensors')
    width, height = int(input_size[0]), int(input_size[1])
    original_height, original_width = int(original_shape[0]), int(original_shape[1])
    if original_height <= 0 or original_width <= 0:
        raise ValueError('original_shape must contain positive dimensions')
    if len(intrinsic) == 4:
        fx, fy, cx, cy = [float(v) for v in intrinsic]
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy],
                                  [0.0, 0.0, 1.0]], dtype=np.float32)
    elif len(intrinsic) == 9:
        camera_matrix = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
    else:
        raise ValueError('intrinsic must contain 4 or 9 values')

    cls_logits = _as_numpy(outputs[0])[0]
    bbox_preds = _as_numpy(outputs[1])[0]
    centers_2d_preds = _as_numpy(outputs[2])[0]
    z_preds = _as_numpy(outputs[3])[0]
    rotation_preds = _as_numpy(outputs[4])[0]
    sizes_preds = _as_numpy(outputs[5])[0]

    # Sigmoid classification followed by the same flattened top-k selection as
    # the PyTorch head (query index = flattened index // class count).
    cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_logits, -80.0, 80.0)))
    flat_scores = cls_scores.reshape(-1)
    count = min(int(max_per_img), flat_scores.size)
    order = np.argsort(-flat_scores, kind='stable')[:count]
    det_labels = order % int(num_classes)
    bbox_index = order // int(num_classes)
    scores = flat_scores[order]

    bbox = bbox_preds[bbox_index].copy()
    centers = centers_2d_preds[bbox_index].copy()
    z = z_preds[bbox_index].copy()
    rotation = rotation_preds[bbox_index].copy()
    sizes = sizes_preds[bbox_index].copy()

    if classwise_rotation:
        rotation = rotation.reshape(count, int(num_classes), -1)
        rotation = rotation[np.arange(count), det_labels]
    if classwise_sizes:
        sizes = sizes.reshape(count, int(num_classes), 3)
        sizes = sizes[np.arange(count), det_labels]

    # cxcywh normalized coordinates -> clamped resized-image xyxy/center.
    bboxes = np.empty((count, 4), dtype=np.float32)
    bboxes[:, 0] = bbox[:, 0] - bbox[:, 2] * 0.5
    bboxes[:, 1] = bbox[:, 1] - bbox[:, 3] * 0.5
    bboxes[:, 2] = bbox[:, 0] + bbox[:, 2] * 0.5
    bboxes[:, 3] = bbox[:, 1] + bbox[:, 3] * 0.5
    bboxes[:, 0::2] *= width
    bboxes[:, 1::2] *= height
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height)

    centers *= np.array([width, height], dtype=np.float32)
    centers[:, 0] = np.clip(centers[:, 0], 0, width)
    centers[:, 1] = np.clip(centers[:, 1], 0, height)

    scale_x = float(width) / float(original_width)
    scale_y = float(height) / float(original_height)
    bboxes[:, 0::2] /= scale_x
    bboxes[:, 1::2] /= scale_y
    centers[:, 0] /= scale_x
    centers[:, 1] /= scale_y

    depth = np.exp(z) if use_log_z else z
    centers_h = np.concatenate(
        [centers, np.ones((count, 1), dtype=np.float32)], axis=1)
    rays = centers_h @ np.linalg.inv(camera_matrix).T
    translations = rays * depth

    return {
        'scores': scores.astype(np.float32),
        'labels': det_labels.astype(np.int64),
        'bboxes': bboxes.astype(np.float32),
        'centers_2d': centers.astype(np.float32),
        'z': z.astype(np.float32),
        'translations': translations.astype(np.float32),
        'rotations': rotation.astype(np.float32),
        'sizes': sizes.astype(np.float32),
    }
