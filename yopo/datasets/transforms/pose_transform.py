import math
from typing import Optional
import warnings

import mmcv
import numpy as np
from numpy import random
import cv2

from yopo.structures.bbox import HorizontalBoxes, BaseBoxes, autocast_box_type
from mmcv.transforms.utils import cache_randomness
from yopo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from yopo.datasets.transforms.transforms import Resize


@TRANSFORMS.register_module()
class RandomAffinefor6DPose(BaseTransform):
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: tuple[float, float] = (0.9, 1.1),
                 max_shear_degree: float = 0.0,
                 border: tuple[int, int] = (0, 0),
                 border_val: tuple[int, int, int] = (114, 114, 114),
                 bbox_clip_border: bool = True) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _get_affine_matrix(
        self,
        image_size: tuple[int, int],
        camera_matrix= None
    ):
        if camera_matrix is None:
            raise ValueError("camera_matrix should not be None")

        img_width, img_height = image_size

        # Rotation and Scale
        angle = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        min_scale = self.scaling_ratio_range[0]
        max_scale = self.scaling_ratio_range[1]
        scale = random.uniform(min_scale, max_scale)

        if scale <= 0.0:
            raise ValueError("Argument scale should be positive")
        center = (camera_matrix[2], camera_matrix[5])
        R = cv2.getRotationMatrix2D(angle=angle, center=center, scale=scale) #Rotate around the principle axis

        M = np.ones([2, 3])
        # Shear
        rand_shear = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_x = math.tan(rand_shear * math.pi / 180)
        shear_y = math.tan(rand_shear * math.pi / 180)

        M[0] = R[0] + shear_y * R[1]
        M[1] = R[1] + shear_x * R[0]

        # Translation
        translation_x = random.uniform(-self.max_translate_ratio,
                                       self.max_translate_ratio) * img_width  # x translation (pixels)
        translation_y = random.uniform(-self.max_translate_ratio,
                                       self.max_translate_ratio) * img_height  # y translation (pixels)

        M[0, 2] += translation_x
        M[1, 2] += translation_y

        return M, scale, angle

    def apply_affine_to_targets(self,
                                results,
                                image_size: tuple[int, int],
                                warp_matrix,
                                scale: float,
                                angle: float):
        num_gts = len(results['gt_bboxes'])
        # warp object center points [tx, ty]
        target_kpts = np.ones((num_gts, 3), dtype=np.float32)
        target_kpts[:, :2] = results['center_2d']
        target_kpts = target_kpts @ warp_matrix.T  # transform
        results['center_2d'] = target_kpts.astype(np.float32)
        #transform Rotation
        rotation_mat = results['T'][:, :, :3]
        deltaR = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=1.0)
        deltaR = np.vstack( (deltaR, np.array([[0, 0, 1.0]])) )
        rotation_mat = deltaR @ rotation_mat
        r1 = rotation_mat[:, :, 0]
        r2 = rotation_mat[:, :, 1]
        r = np.concatenate((r1, r2), axis=1).reshape(num_gts, 6).astype(np.float32)
        results['rotation'] = r
        # transform depth
        results['z'] = np.log(np.exp(results['z']) / scale)
        # results['z'] = results['z'] / scale
        return results

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        intrinsic = results['intrinsic']
        warp_matrix, scale, angle = self._get_affine_matrix((width, height),
                                                            camera_matrix=intrinsic)

        img = cv2.warpAffine(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)

        results['img'] = img
        results['img_shape'] = img.shape[:2]

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            # make the warp matrix 3x3
            homography_matrix = np.vstack((warp_matrix, [0, 0, 1]))
            bboxes.project_(homography_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]
            
            results['rotation'] = results['rotation'][valid_index]
            results['center_2d'] = results['center_2d'][valid_index]
            results['z'] = results['z'][valid_index]
            results['T'] = results['T'][valid_index]

            if 'gt_masks' in results:
                raise NotImplementedError('RandomAffine only supports bbox.')
        

        results = self.apply_affine_to_targets(
            results, img.shape[:2], warp_matrix, scale, angle)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio_range={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float,
                          y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix


@TRANSFORMS.register_module()
class RandomTranslatePixels(BaseTransform):
    """Translate image, bounding boxes and camera intrinsics.
    Args:
        prob (float): Probability of translating.
        max_translate_offset (int): Maximum pixel offset for translation.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Defaults to 1.
        pad_val (int): Padding value. Defaults to 0.
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_translate_offset: int = 50,
                 filter_thr_px: int = 1,
                 pad_val: int = 0,
                 shift_depth: bool = False) -> None:
        assert 0 <= prob <= 1
        self.prob = prob
        self.max_translate_offset = max_translate_offset
        self.filter_thr_px = filter_thr_px
        self.pad_val = pad_val
        self.shift_depth = shift_depth

    @cache_randomness
    def _get_offset(self) -> tuple[int, int]:
        if random.random() < self.prob:
            offset_x = random.randint(-self.max_translate_offset,
                                      self.max_translate_offset)
            offset_y = random.randint(-self.max_translate_offset,
                                      self.max_translate_offset)
            return offset_x, offset_y
        return 0, 0

    @autocast_box_type()
    def transform(self, results: dict) -> Optional[dict]:
        """
        Required Keys:
        - img
        - gt_bboxes
        - intrinsic
        - translation (optional)
        - T (optional)
        - center_2d (optional)
        Modified Keys:
        - img
        - gt_bboxes
        - intrinsic
        - translation
        - T
        - center_2d
        - other gt_* fields
        """
        offset_x, offset_y = self._get_offset()

        if offset_x == 0 and offset_y == 0:
            return results

        img = results['img']
        img_h, img_w = img.shape[:2]

        # shift img
        new_img = np.full_like(img, self.pad_val)
        new_x = 0 if offset_x > 0 else -offset_x
        new_y = 0 if offset_y > 0 else -offset_y
        ori_x = offset_x if offset_x > 0 else 0
        ori_y = offset_y if offset_y > 0 else 0
        new_h = img_h - np.abs(offset_y)
        new_w = img_w - np.abs(offset_x)

        if new_h <= 0 or new_w <= 0:
            return results

        new_img[new_y:new_y + new_h, new_x:new_x + new_w] = \
            img[ori_y:ori_y + new_h, ori_x:ori_x + new_w]
        results['img'] = new_img

        if self.shift_depth:
            if 'depth' not in results or results['depth'] is None:
                raise ValueError('shift_depth is True, but depth is not found')
            depth = results['depth']
            max_distance = np.max(depth)
            new_depth = np.full_like(depth, max_distance)
            new_depth[new_y:new_y + new_h, new_x:new_x + new_w] = \
                depth[ori_y:ori_y + new_h, ori_x:ori_x + new_w]
            results['depth'] = new_depth

        # --- Update 3D Poses ---
        # Get camera intrinsics to back-project 2D offset to 3D
        K = None
        if 'intrinsic' in results:
            intrinsic_val = results['intrinsic']
            if isinstance(intrinsic_val, list) and len(intrinsic_val) == 4:
                K = np.array([[intrinsic_val[0], 0, intrinsic_val[2]],
                              [0, intrinsic_val[1], intrinsic_val[3]],
                              [0, 0, 1]], dtype=np.float32)
            elif isinstance(intrinsic_val, list) and not isinstance(
                    intrinsic_val[0], (list, np.ndarray)):
                K = np.array([[intrinsic_val[0], 0, intrinsic_val[2]],
                              [0, intrinsic_val[1], intrinsic_val[3]],
                              [0, 0, 1]], dtype=np.float32)
            elif isinstance(intrinsic_val, list):
                K = intrinsic_val[0]
            else:
                K = intrinsic_val

        if K is not None:
            fx, fy = K[0, 0], K[1, 1]
            if 'translation' in results and len(results['translation']) > 0:
                z = results['translation'][:, 2]
                # Use abs() to handle potential negative depth values
                z_abs = np.abs(z)
                delta_x_3d = (offset_x * z_abs) / fx
                delta_y_3d = (offset_y * z_abs) / fy

                results['translation'][:, 0] += delta_x_3d
                results['translation'][:, 1] += delta_y_3d

                if 'T' in results:
                    results['T'][:, 0, 3] = results['translation'][:, 0]
                    results['T'][:, 1, 3] = results['translation'][:, 1]

            if 'center_2d' in results and len(results['center_2d']) > 0:
                results['center_2d'][:, 0] += offset_x
                results['center_2d'][:, 1] += offset_y

        # Translate bounding boxes
        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            num_bboxes = len(bboxes)
            bboxes.translate_([offset_x, offset_y])
            bboxes.clip_([img_h, img_w])

            if self.filter_thr_px > 0:
                valid_inds = (bboxes.widths > self.filter_thr_px) & \
                             (bboxes.heights > self.filter_thr_px)

                if not valid_inds.all():
                    valid_inds_np = valid_inds.cpu().numpy()
                    results['gt_bboxes'] = bboxes[valid_inds]

                    # Explicitly filter all related annotations
                    keys_to_filter = [
                        'gt_bboxes_labels', 'gt_ignore_flags',
                        'translation', 'rotation', 'size', 'center_2d', 'z', 'T'
                    ]

                    for key in keys_to_filter:
                        if key in results:
                            field = results[key]
                            if isinstance(field, (np.ndarray, list)) and len(field) == num_bboxes:
                                results[key] = np.array(field)[valid_inds_np]

                    # also filter 'instances' list of dicts
                    if 'instances' in results and len(results['instances']) == num_bboxes:
                        results['instances'] = [
                            inst for i, inst in
                            enumerate(results['instances']) if valid_inds_np[i]
                        ]

                if len(results['gt_bboxes']) == 0:
                    return None

        # Translate camera intrinsics
        if 'intrinsic' in results:
            K_res = results['intrinsic']
            if isinstance(K_res, list) and len(K_res) == 4:
                K_res[2] += offset_x
                K_res[3] += offset_y
            elif isinstance(K_res, list) and not isinstance(
                    K_res[0], (list, np.ndarray)):
                K_res[2] += offset_x
                K_res[3] += offset_y
            elif isinstance(K_res, list):
                new_intrinsics = []
                for K_mat in K_res:
                    K_new = K_mat.copy()
                    K_new[0, 2] += offset_x
                    K_new[1, 2] += offset_y
                    new_intrinsics.append(K_new)
                results['intrinsic'] = new_intrinsics
            elif isinstance(K_res, np.ndarray):
                K_res[0, 2] += offset_x
                K_res[1, 2] += offset_y

        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(prob={self.prob}, ' \
               f'max_translate_offset={self.max_translate_offset}, ' \
               f'filter_thr_px={self.filter_thr_px})'

@TRANSFORMS.register_module()
class ResizeOBBGaussians(BaseTransform):
    """Apply the current image resize affine map to OBB Gaussians.

    This transform must run immediately after the image/bbox ``Resize``. The
    compact Gaussian layout is ``(cx, cy, covariance_xx, covariance_xy,
    covariance_yy)``.
    """

    def transform(self, results: dict) -> dict:
        if 'obb_gaussian' not in results:
            return results
        if 'scale_factor' not in results:
            raise KeyError(
                'ResizeOBBGaussians requires scale_factor from Resize')
        scale_factor = np.asarray(results['scale_factor'], dtype=np.float32)
        if scale_factor.size < 2:
            raise ValueError(
                f'scale_factor must contain x/y scales, got {scale_factor}')
        scale_x, scale_y = float(scale_factor[0]), float(scale_factor[1])
        gaussians = np.asarray(
            results['obb_gaussian'], dtype=np.float32).copy()
        if gaussians.ndim != 2 or gaussians.shape[1] != 5:
            raise ValueError(
                'obb_gaussian must have shape (N, 5), got '
                f'{gaussians.shape}')
        gaussians[:, 0] *= scale_x
        gaussians[:, 1] *= scale_y
        gaussians[:, 2] *= scale_x * scale_x
        gaussians[:, 3] *= scale_x * scale_y
        gaussians[:, 4] *= scale_y * scale_y
        results['obb_gaussian'] = gaussians
        return results


@TRANSFORMS.register_module()
class RandomFlipFor9DPose(BaseTransform):
    """Flip the image and 9D pose annotations.

    Required Keys:
        - img
        - gt_bboxes (optional)
        - intrinsic
        - translation (optional)
        - rotation (optional)
        - T (optional)
        - center_2d (optional)

    Modified Keys:
        - img
        - gt_bboxes
        - intrinsic
        - translation
        - rotation
        - T
        - center_2d

    Args:
        prob (float): The flipping probability. Defaults to 0.5.
        direction (str): The flipping direction. Defaults to 'horizontal'.
            Currently only 'horizontal' is supported for 9D pose annotations.
    """

    def __init__(self, prob: float = 0.5, direction: str = 'horizontal'):
        if direction not in ['horizontal', 'vertical']:
            raise ValueError(f'Direction {direction} is not supported.')
        if direction != 'horizontal':
            warnings.warn(
                'Vertical flip for 9D pose is not implemented. '
                'Only image and bboxes will be flipped.')
        assert 0 <= prob <= 1
        self.prob = prob
        self.direction = direction

    @cache_randomness
    def _get_flip_flag(self) -> bool:
        """A function to determine whether to flip."""
        return random.random() < self.prob

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bboxes and pose annotations.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results.
        """
        is_flip = self._get_flip_flag()
        if not is_flip:
            return results

        # flip image
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        img_h, img_w = results['img'].shape[:2]

        # flip bboxes
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            results['gt_bboxes'].flip_((img_h, img_w), direction=self.direction)

        if self.direction != 'horizontal':
            return results

        # flip 9D pose annotations for horizontal flip
        # flip intrinsic
        if 'intrinsic' in results:
            K = results['intrinsic']
            if isinstance(K, list) and len(K) == 4:  # [fx, fy, cx, cy]
                K[2] = img_w - 1 - K[2]
            elif isinstance(K, np.ndarray) and K.shape == (3, 3):
                K[0, 2] = img_w - 1 - K[0, 2]
            results['intrinsic'] = K

        # flip translation
        if 'translation' in results and len(results['translation']) > 0:
            results['translation'][:, 0] *= -1

        # flip rotation
        if 'rotation' in results and len(results['rotation']) > 0:
            # 6D representation: [c1, c2] where c1, c2 are first two columns of R
            # For horizontal flip, R' = diag([-1, 1, 1]) @ R.
            # This negates the first row of R.
            results['rotation'][:, 0] *= -1
            results['rotation'][:, 3] *= -1

        # flip center_2d
        if 'center_2d' in results and len(results['center_2d']) > 0:
            results['center_2d'][:, 0] = img_w - 1 - results['center_2d'][:, 0]

        # Reflect the OBB Gaussian. F=diag(-1, 1) preserves xx/yy and
        # negates only the xy covariance, avoiding angle-wrap conventions.
        if 'obb_gaussian' in results and len(results['obb_gaussian']) > 0:
            results['obb_gaussian'][:, 0] = (
                img_w - 1 - results['obb_gaussian'][:, 0])
            results['obb_gaussian'][:, 3] *= -1

        # flip T (4x4 transformation matrix)
        if 'T' in results and len(results['T']) > 0:
            # t' = M @ t
            results['T'][:, 0, 3] *= -1
            # R' = M @ R
            results['T'][:, 0, :3] *= -1

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'direction={self.direction})')

@TRANSFORMS.register_module()
class RandomRotationFor9DPose(BaseTransform):
    """Randomly rotate the image and update pose annotations.

    This transform rotates the image around its center. The 3D pose of
    each object is updated accordingly. Bounding boxes are recomputed
    by projecting the 3D model corners.

    Required Keys:
    - img
    - intrinsic
    - T
    - size
    - gt_bboxes_labels
    - translation (optional)
    - rotation (optional)
    - center_2d (optional)
    - z (optional)

    Modified Keys:
    - img
    - gt_bboxes
    - T
    - translation
    - rotation
    - center_2d
    - z

    Args:
        prob (float): Probability of applying this transform. Defaults to 0.5.
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        use_log_z (bool): Whether to use log scale for z. Defaults to False.
        sym_ids (list[int]): List of class ids for symmetric objects.
            The rotation of these objects will be canonicalized.
            Defaults to `[0, 1, 3]` for NOCS.
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_rotate_degree: float = 10.0,
                 use_log_z: bool = False,
                 sym_ids: list = [0, 1, 3]) -> None:
        if not (0 <= prob <= 1):
            raise ValueError(f'Probability {prob} is not in [0, 1].')
        self.prob = prob
        self.max_rotate_degree = max_rotate_degree
        self.use_log_z = use_log_z
        self.sym_ids = sym_ids


    @cache_randomness
    def _get_rotate_flag(self) -> bool:
        """A function to determine whether to rotate."""
        return random.random() < self.prob

    @cache_randomness
    def _get_random_rotation_info(self):
        """Get random rotation matrix and degree."""
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        radian = math.radians(rotation_degree)
        # rotation around z-axis
        R_aug = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return R_aug, rotation_degree

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to randomly rotate images, bounding boxes and
        pose annotations."""
        if not self._get_rotate_flag():
            return results

        img = results['img']
        h, w = img.shape[:2]

        R_aug, rotation_degree = self._get_random_rotation_info()

        # get 2D rotation matrix for image
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_degree, 1)

        # rotate image
        img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
        results['img'] = img

        if 'depth' in results and results['depth'] is not None:
            # also rotate depth image if it exists
            results['depth'] = cv2.warpAffine(
                results['depth'],
                M, (w, h),
                borderValue=0,
                flags=cv2.INTER_NEAREST)

        # update annotations
        if 'T' in results and len(results['T']) > 0:
            intrinsic = results['intrinsic']
            if isinstance(intrinsic, list) and len(intrinsic) == 4:
                K = np.array([[intrinsic[0], 0, intrinsic[2]],
                              [0, intrinsic[1], intrinsic[3]],
                              [0, 0, 1]],
                             dtype=np.float32)
            elif isinstance(intrinsic,
                            (list, np.ndarray)) and np.array(intrinsic).size == 9:
                K = np.array(intrinsic).reshape(3, 3)
            else:
                raise ValueError(
                    f'Invalid intrinsic shape: {np.array(intrinsic).shape}')

            num_instances = len(results['T'])
            for i in range(num_instances):
                # original pose
                T = results['T'][i]
                R = T[:3, :3]
                t = T[:3, 3]

                # new pose
                R_new = R_aug @ R
                t_new = R_aug @ t

                # symmetry handling for symmetric objects
                class_id = results['gt_bboxes_labels'][i]
                if class_id in self.sym_ids:
                    theta_x = R_new[0, 0] + R_new[2, 2]
                    theta_y = R_new[0, 2] - R_new[2, 0]
                    r_norm = np.sqrt(theta_x**2 + theta_y**2)
                    if r_norm > 1e-6:
                        s_map = np.array(
                            [[theta_x / r_norm, 0.0, -theta_y / r_norm],
                             [0.0, 1.0, 0.0],
                             [theta_y / r_norm, 0.0, theta_x / r_norm]],
                            dtype=np.float32)
                        R_new = R_new @ s_map

                # update instance annotations
                results['T'][i, :3, :3] = R_new
                results['T'][i, :3, 3] = t_new
                if 'translation' in results:
                    results['translation'][i] = t_new
                if 'rotation' in results:
                    results['rotation'][i] = [
                        R_new.flatten()[j] for j in [0, 3, 6, 1, 4, 7]
                    ]

                if 'center_2d' in results:
                    center_2d = K @ t_new / t_new[2]
                    results['center_2d'][i] = center_2d[:2]

                if 'z' in results:
                    if self.use_log_z:
                        results['z'][i] = np.log(t_new[2])
                    else:
                        results['z'][i] = t_new[2]

            # update gt_bboxes by rotating 2D bboxes
            if 'gt_bboxes' in results:
                bboxes = results['gt_bboxes'].tensor.numpy()
                new_bboxes = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1],
                                        [x2, y2, 1]]).T
                    rotated_corners = M @ corners
                    x_min, y_min = rotated_corners.min(axis=1)
                    x_max, y_max = rotated_corners.max(axis=1)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    new_bboxes.append([x_min, y_min, x_max, y_max])

                if len(new_bboxes) > 0:
                    results['gt_bboxes'] = HorizontalBoxes(
                        np.array(new_bboxes, dtype=np.float32))
                else:
                    results['gt_bboxes'] = HorizontalBoxes(
                        np.zeros((0, 4), dtype=np.float32))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'use_log_z={self.use_log_z}, '
        repr_str += f'sym_ids={self.sym_ids})'
        return repr_str


@TRANSFORMS.register_module()
class ResizeforPose(Resize):
    """Resize the image and update pose annotations.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor. Pose-related annotations like intrinsics
    and 2d centers are also updated. 3D annotations like ``translation``,
    ``T`` (pose matrix), and ``size`` are not affected by this transform.

    Required Keys:
    - img
    - intrinsic
    - gt_bboxes (optional)
    - translation (optional)
    - T (optional)
    - center_2d (optional)
    - size (optional)

    Modified Keys:
    - img
    - img_shape
    - intrinsic
    - gt_bboxes
    - center_2d

    Added Keys:
    - scale
    - scale_factor
    - keep_ratio
    """

    def _resize_pose(self, results: dict) -> None:
        """Resize pose-related annotations."""
        scale_factor = results['scale_factor']
        # preserve original intrinsic parameter
        # since the center_2d is going to restored in the original image size

        # update center_2d
        if 'center_2d' in results and len(results['center_2d']) > 0:
            results['center_2d'][:, 0] *= scale_factor[0]
            results['center_2d'][:, 1] *= scale_factor[1]

        # 3D properties like translation, T, and size are not affected by
        # image resizing.

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and
        pose annotations."""
        results = super().transform(results)
        if results is None:
            return None
        self._resize_pose(results)
        return results

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        return repr_str



@TRANSFORMS.register_module()
class PadAndResizeForPoseTest(Resize):
    """Resize the image and update pose annotations.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor. Pose-related annotations like intrinsics
    and 2d centers are also updated. 3D annotations like ``translation``,
    ``T`` (pose matrix), and ``size`` are not affected by this transform.

    Required Keys:
    - img
    - intrinsic
    - gt_bboxes (optional)
    - translation (optional)
    - T (optional)
    - center_2d (optional)
    - size (optional)

    Modified Keys:
    - img
    - img_shape
    - intrinsic
    - gt_bboxes
    - center_2d

    Added Keys:
    - scale
    - scale_factor
    - keep_ratio
    """

    def __init__(self, *args, original_img_shape=None, scale_factor=1., **kwargs):
        super().__init__(*args, scale_factor=scale_factor, **kwargs)
        if original_img_shape is None:
            raise ValueError(
                'original_img_shape must be provided for PadAndResizeForPoseTest')
        self.original_img_shape = original_img_shape

    def _resize_pose(self, results: dict) -> None:
        """Resize pose-related annotations."""
        scale_factor = results['scale_factor']
        # preserve original intrinsic parameter
        # since the center_2d is going to restored in the original image size

        # update center_2d
        if 'center_2d' in results and len(results['center_2d']) > 0:
            results['center_2d'][:, 0] *= scale_factor[0]
            results['center_2d'][:, 1] *= scale_factor[1]

        # 3D properties like translation, T, and size are not affected by
        # image resizing.

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and
        pose annotations."""

        target_aspect = self.original_img_shape[0] / self.original_img_shape[1]

        target_width = int(results['img_shape'][0] * target_aspect)

        pad_amount = (target_width - results['img_shape'][1])
        pad_left = pad_amount // 2

        padded_image = np.zeros((results['img_shape'][0], target_width, 3), dtype=np.uint8)
        padded_image[:, pad_left:pad_left + results['img_shape'][1]] = results['img']
        
        # update intrinsic
        if 'intrinsic' in results:
            K = results['intrinsic']
            if isinstance(K, list) and len(K) == 4:
                K[2] += pad_left
        
        pad_height, pad_width = padded_image.shape[:2]
        
        x_scale = self.original_img_shape[0] / pad_width
        y_scale = self.original_img_shape[1] / pad_height

        self.scale_factor = (x_scale, y_scale)

        results['img'] = padded_image
        results['img_shape'] = padded_image.shape[:2]
        
        results = super().transform(results)
        if results is None:
            return None
        self._resize_pose(results)
        # update the intrinsic
        if isinstance(K, list) and len(K) == 4:
            K[0] *= x_scale
            K[1] *= y_scale
            K[2] *= x_scale
            K[3] *= y_scale
        results['intrinsic'] = K
        return results

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        return repr_str
