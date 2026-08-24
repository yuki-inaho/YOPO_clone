# Copyright (c) OpenMMLab. All rights reserved.
import math
from abc import abstractmethod
from typing import Optional, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from yopo.registry import TASK_UTILS
from yopo.structures.bbox import (BaseBoxes, bbox_overlaps,
                                  bbox_xyxy_to_cxcywh, rbbox_overlaps)
from yopo.models.losses.gaussian_dist_loss import (
    postprocess, xy_stddev_pearson_2_xy_sigma, xy_wh_r_2_xy_sigma)


class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            img_meta (dict, optional): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass


@TASK_UTILS.register_module()
class BBoxL1Cost(BaseMatchCost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from yopo.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 box_format: str = 'xyxy',
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # convert box format
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            pred_bboxes = bbox_xyxy_to_cxcywh(pred_bboxes)

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class IoUCost(BaseMatchCost):
    """IoUCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'giou'. Defaults to 'giou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from yopo.models.task_modules.assigners.
        ... match_costs.match_cost import IoUCost
        >>> import torch
        >>> self = IoUCost()
        >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
        >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> self(bboxes, gt_bboxes)
        tensor([[-0.1250,  0.1667],
            [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode: str = 'giou', weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # avoid fp16 overflow
        if pred_bboxes.dtype == torch.float16:
            fp16 = True
            pred_bboxes = pred_bboxes.to(torch.float32)
        else:
            fp16 = False

        overlaps = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        if fp16:
            overlaps = overlaps.to(torch.float16)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


@TASK_UTILS.register_module()
class ClassificationCost(BaseMatchCost):
    """ClsSoftmaxCost.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from yopo.models.task_modules.assigners.
        ...  match_costs.match_cost import ClassificationCost
        >>> import torch
        >>> self = ClassificationCost()
        >>> cls_pred = torch.rand(4, 3)
        >>> gt_labels = torch.tensor([0, 1, 2])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(cls_pred, gt_labels)
        tensor([[-0.3430, -0.3525, -0.3045],
            [-0.3077, -0.2931, -0.3992],
            [-0.3664, -0.3455, -0.2881],
            [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``scores`` inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (:obj:`InstanceData`): ``labels`` inside should have
                shape (num_gt, ).
            img_meta (Optional[dict]): _description_. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels

        pred_scores = pred_scores.softmax(-1)
        cls_cost = -pred_scores[:, gt_labels]

        return cls_cost * self.weight


@TASK_UTILS.register_module()
class FocalLossCost(BaseMatchCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 eps: float = 1e-12,
                 binary_input: bool = False,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
                in shape (num_queries, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_queries, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        if self.binary_input:
            pred_masks = pred_instances.masks
            gt_masks = gt_instances.masks
            return self._mask_focal_loss_cost(pred_masks, gt_masks)
        else:
            pred_scores = pred_instances.scores
            gt_labels = gt_instances.labels
            return self._focal_loss_cost(pred_scores, gt_labels)


@TASK_UTILS.register_module()
class BinaryFocalLossCost(FocalLossCost):

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # gt_instances.text_token_mask is a repeated tensor of the same length
        # of instances. Only gt_instances.text_token_mask[0] is useful
        text_token_mask = torch.nonzero(
            gt_instances.text_token_mask[0]).squeeze(-1)
        pred_scores = pred_instances.scores[:, text_token_mask]
        gt_labels = gt_instances.positive_maps[:, text_token_mask]
        return self._focal_loss_cost(pred_scores, gt_labels)


@TASK_UTILS.register_module()
class DiceCost(BaseMatchCost):
    """Cost of mask assignments based on dice losses.

    Args:
        pred_act (bool): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float): Defaults to 1e-3.
        naive_dice (bool): If True, use the naive dice loss
            in which the power of the number in the denominator is
            the first power. If False, use the second power that
            is adopted by K-Net and SOLO. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 pred_act: bool = False,
                 eps: float = 1e-3,
                 naive_dice: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def _binary_mask_dice_loss(self, mask_preds: Tensor,
                               gt_masks: Tensor) -> Tensor:
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (num_queries, *).
            gt_masks (Tensor): Ground truth in shape (num_gt, *)
                store 0 or 1, 0 for negative class and 1 for
                positive class.

        Returns:
            Tensor: Dice cost matrix in shape (num_queries, num_gt).
        """
        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()
        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + \
                          gt_masks.sum(-1)[None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[:, None] + \
                          gt_masks.pow(2).sum(1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks

        if self.pred_act:
            pred_masks = pred_masks.sigmoid()
        dice_cost = self._binary_mask_dice_loss(pred_masks, gt_masks)
        return dice_cost * self.weight


@TASK_UTILS.register_module()
class CrossEntropyLossCost(BaseMatchCost):
    """CrossEntropyLossCost.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 use_sigmoid: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred: Tensor,
                              gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_queries, 1, *) or
                (num_queries, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_queries, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``masks``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(pred_masks, gt_masks)
        else:
            raise NotImplementedError

        return cls_cost * self.weight

@TASK_UTILS.register_module()
class StableDINOFocalLossCost(FocalLossCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        cec_beta (float): Defaults to 0.5.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 cec_beta: float = 0.5,
                 eps: float = 1e-12,
                 binary_input: bool = False,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(alpha=alpha, gamma=gamma,
                         eps=eps, binary_input=binary_input,
                         weight=weight)
        self.cec_beta = cec_beta

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_scores = pred_instances.scores.sigmoid()
        gt_labels = gt_instances.labels

        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        # assign weights based on giou
        gious = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode='giou', is_aligned=False)
        gious = (gious + 1) / 2
        # rescale max gious to 1
        max_giou = gious.max()
        scalar = 1. / (max_giou + self.eps)
        scalar = max(scalar, 1)
        gious = gious * scalar

        weights = torch.zeros_like(pred_scores)
        weights[:, gt_labels] = gious

        pred_scores = pred_scores * (weights ** self.cec_beta)

        return self._focal_loss_cost(pred_scores, gt_labels)

@TASK_UTILS.register_module()
class ADDCost(BaseMatchCost):
    """ADDCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from yopo.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
    
    def _add_cost(self, pred_instances: InstanceData,
                   gt_instances: InstanceData,
                   img_meta: Optional[dict] = None) -> Tensor:
        """Compute pose cost.
    
        Args:
              pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
               predicted boxes with unnormalized coordinate
               (x, y, x, y).
              gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
               bboxes with unnormalized coordinate (x, y, x, y).
              img_meta (Optional[dict]): Image information. Defaults to None.
    
        Returns:
              Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # compute the ADD cost
        pred_R = pred_instances.rotations
        pred_centers_2d = pred_instances.centers_2d
        pred_z = pred_instances.z

        r1, r2 = torch.split(pred_R, 3, dim=-1)
        r1 = r1 / torch.norm(r1, dim=-1, keepdim=True).clamp_min(1e-6)
        r2 = r2 - torch.bmm(r1.unsqueeze(1),
            r2.unsqueeze(-1)).squeeze(-1) * r1
        r2 = r2 / torch.norm(r2, dim=-1, keepdim=True).clamp_min(1e-6)
        r3 = torch.cross(r1, r2, dim=-1)
        pred_R = torch.stack((r1, r2, r3), dim=-1)

        intrinsic = torch.tensor(img_meta['intrinsic'], dtype=torch.float32).to(pred_R.device)
        intrinsic = intrinsic.reshape(3, 3)

        centers_2d_h = torch.cat((pred_centers_2d, torch.ones_like(pred_centers_2d[..., :1])), dim=-1)
        depth = torch.exp(pred_z)
        # depth = pred_z * 1800
        pred_t = depth * (torch.linalg.inv(intrinsic) @ centers_2d_h.T).T

        gt_T = gt_instances.T
        gt_R = gt_T[:, :3, :3]
        gt_t = gt_T[:, :3, 3]

        # use unit cube for keypoints
        # define the unit cube
        unit_cube = torch.tensor([[-0.5, -0.5, 0],
                                   [0.5, -0.5, 0],
                                   [-0.5, 0.5, 0],
                                   [0.5, 0.5, 0],
                                   [-0.5, -0.5, 1],
                                   [0.5, -0.5, 1],
                                   [-0.5, 0.5, 1],
                                   [0.5, 0.5, 1]],
                                   dtype=torch.float32).to(pred_R.device)
        # transform the unit cube to the predicted pose
        # pred_3d_points = torch.bmm(pred_R, unit_cube.T).T + pred_t
        # gt_3d_points = torch.bmm(gt_R, unit_cube.T).T + gt_t
        pred_3d_points = (pred_R @ unit_cube.T) + pred_t[..., None]
        gt_3d_points = (gt_R @ unit_cube.T) + gt_t[..., None]
        pred_3d_points = pred_3d_points.permute(0, 2, 1)
        gt_3d_points = gt_3d_points.permute(0, 2, 1)

        # compute the distance between the predicted and gt 3d points
        pred_3d_points = pred_3d_points.unsqueeze(1)
        gt_3d_points = gt_3d_points.unsqueeze(0)

        pose_cost = torch.norm(pred_3d_points - gt_3d_points, dim=2).mean(dim=-1)

        # normalize with the maximum distance
        pose_cost = pose_cost / 1000  # from mm to m
        return pose_cost

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        cost = self._add_cost(pred_instances, gt_instances, img_meta)
        return cost * self.weight
    

@TASK_UTILS.register_module()
class DepthCost(BaseMatchCost):
    """DepthCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from yopo.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 box_format: str = 'xyxy',
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_z = pred_instances.z
        gt_z = gt_instances.z

        pred_z = torch.exp(pred_z)
        gt_z = torch.exp(gt_z)

        z_cost = torch.cdist(pred_z, gt_z, p=2)
        return z_cost * self.weight

@TASK_UTILS.register_module()
class PoseL2Cost(BaseMatchCost):
    """PoseCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
        symmetric_class_ids (list[int], optional): A list of class ids that
            are symmetric. Defaults to None.

    Examples:
        >>> from yopo.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 weight: Union[float, int] = 1.,
                 symmetric_class_ids: list[int] = None) -> None:
        super().__init__(weight=weight)
        self.symmetric_class_ids = symmetric_class_ids if symmetric_class_ids is not None else []

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_R = pred_instances.rotations
        pred_centers_2d = pred_instances.centers_2d
        pred_z = pred_instances.z

        gt_R = gt_instances.rotations
        gt_centers_2d = gt_instances.centers_2d
        gt_z = gt_instances.z
        gt_labels = gt_instances.labels

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_centers_2d.new_tensor([img_w, img_h]).unsqueeze(0)
        pred_centers_2d = pred_centers_2d / factor
        gt_centers_2d = gt_centers_2d / factor

        R_cost = torch.cdist(pred_R, gt_R, p=2)

        if self.symmetric_class_ids:
            is_symmetric = torch.zeros_like(gt_labels, dtype=torch.bool)
            for class_id in self.symmetric_class_ids:
                is_symmetric = is_symmetric | (gt_labels == class_id)

            if torch.any(is_symmetric):
                gt_R_sym = gt_R.clone()
                gt_R_sym[is_symmetric, :3] = -gt_R_sym[is_symmetric, :3]
                R_cost_sym = torch.cdist(pred_R, gt_R_sym, p=2)
                R_cost = torch.min(R_cost, R_cost_sym)

        centers_2d_cost = torch.cdist(pred_centers_2d, gt_centers_2d, p=1)
        z_cost = torch.cdist(pred_z, gt_z, p=2)

        pose_cost = R_cost + centers_2d_cost + z_cost
        return pose_cost * self.weight


@TASK_UTILS.register_module()
class IoU3DCost(BaseMatchCost):
    """3D IoU Cost.

    This cost calculates the 3D IoU between predicted and gt bounding boxes.
    The 3D bounding boxes are calculated from the 8 corners of a cube
    transformed by the predicted and ground truth poses and sizes.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances.
                It must contain ``rotations``, ``centers_2d``, ``z``,
                and ``sizes``.
            gt_instances (:obj:`InstanceData`): Ground truth instances.
                It must contain ``T`` (transformation matrix) and ``sizes``.
            img_meta (dict, optional): Image information, which should contain
                ``intrinsic``.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_R = pred_instances.rotations
        pred_centers_2d = pred_instances.centers_2d
        pred_z = pred_instances.z
        pred_sizes = pred_instances.sizes

        # Reconstruct predicted rotation matrix to be orthonormal
        r1, r2 = torch.split(pred_R, 3, dim=-1)
        r1 = r1 / torch.norm(r1, dim=-1, keepdim=True).clamp_min(1e-6)
        r2 = r2 - torch.bmm(r1.unsqueeze(1),
                            r2.unsqueeze(-1)).squeeze(-1) * r1
        r2 = r2 / torch.norm(r2, dim=-1, keepdim=True).clamp_min(1e-6)
        r3 = torch.cross(r1, r2, dim=-1)
        pred_R = torch.stack((r1, r2, r3), dim=-1)

        intrinsic = torch.tensor(img_meta['intrinsic'], dtype=torch.float32).to(pred_R.device)
        intrinsic = intrinsic.reshape(3, 3)

        centers_2d_h = torch.cat((pred_centers_2d, torch.ones_like(pred_centers_2d[..., :1])), dim=-1)
        depth = torch.exp(pred_z)
        pred_t = depth * (torch.linalg.inv(intrinsic) @ centers_2d_h.T).T

        gt_T = gt_instances.T
        gt_R = gt_T[:, :3, :3]
        gt_t = gt_T[:, :3, 3]
        gt_sizes = gt_instances.sizes

        # Define the unit cube centered at the origin
        unit_cube = torch.tensor([[-0.5, -0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [-0.5, 0.5, -0.5],
                                  [0.5, 0.5, -0.5],
                                  [-0.5, -0.5, 0.5],
                                  [0.5, -0.5, 0.5],
                                  [-0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]],
                                 dtype=torch.float32).to(pred_R.device)

        # Scale the unit cube by the sizes to get the corners
        pred_corners = unit_cube.unsqueeze(0) * pred_sizes.unsqueeze(1)
        gt_corners = unit_cube.unsqueeze(0) * gt_sizes.unsqueeze(1)

        # Transform the corners to the predicted and gt poses
        pred_3d_points = torch.bmm(pred_R, pred_corners.permute(0, 2, 1)) + pred_t.unsqueeze(-1)
        gt_3d_points = torch.bmm(gt_R, gt_corners.permute(0, 2, 1)) + gt_t.unsqueeze(-1)
        pred_3d_points = pred_3d_points.permute(0, 2, 1)
        gt_3d_points = gt_3d_points.permute(0, 2, 1)

        # Get AABB from 3d points
        pred_min_coords, _ = torch.min(pred_3d_points, dim=1)
        pred_max_coords, _ = torch.max(pred_3d_points, dim=1)
        pred_bboxes_3d = torch.cat([pred_min_coords, pred_max_coords], dim=1)

        gt_min_coords, _ = torch.min(gt_3d_points, dim=1)
        gt_max_coords, _ = torch.max(gt_3d_points, dim=1)
        gt_bboxes_3d = torch.cat([gt_min_coords, gt_max_coords], dim=1)

        # Expand dims to compute for all pairs
        pred_bboxes_3d = pred_bboxes_3d.unsqueeze(1)  # (num_preds, 1, 6)
        gt_bboxes_3d = gt_bboxes_3d.unsqueeze(0)    # (1, num_gts, 6)

        # Intersection
        inter_min_coords = torch.max(pred_bboxes_3d[..., :3], gt_bboxes_3d[..., :3])
        inter_max_coords = torch.min(pred_bboxes_3d[..., 3:], gt_bboxes_3d[..., 3:])

        inter_whd = (inter_max_coords - inter_min_coords).clamp(min=0)
        inter_vol = inter_whd[..., 0] * inter_whd[..., 1] * inter_whd[..., 2]

        # Union
        pred_whd = (pred_bboxes_3d[..., 3:] - pred_bboxes_3d[..., :3]).clamp(min=0)
        pred_vol = pred_whd[..., 0] * pred_whd[..., 1] * pred_whd[..., 2]

        gt_whd = (gt_bboxes_3d[..., 3:] - gt_bboxes_3d[..., :3]).clamp(min=0)
        gt_vol = gt_whd[..., 0] * gt_whd[..., 1] * gt_whd[..., 2]

        union_vol = pred_vol + gt_vol - inter_vol

        iou = inter_vol / (union_vol + 1e-8)
        iou_cost = 1 - iou

        return iou_cost * self.weight


@TASK_UTILS.register_module()
class TranslationCost(BaseMatchCost):
    """Compute the euclidean distance between the predicted and ground truth for the translation.
    Note the metric unit is in centimeters.
    
    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """
    def __init__(self, weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.
        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``translations``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``translations``.
            img_meta (Optional[dict]): Image information. Defaults to None.
    
        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_translations = pred_instances.translations
        gt_translations = gt_instances.translations

        shift_cost_in_m = torch.cdist(pred_translations, gt_translations, p=2)
        cost = shift_cost_in_m # * 100  # convert to cm
        return cost * self.weight

@TASK_UTILS.register_module()
class RotationCost(BaseMatchCost):
    """Compute the angle difference between the predicted and ground truth for the rotation.
    The cost unit is in degrees.
    Support the y-axis symmetric rotation.

    Args:
        symmetric_classes (list[int]): The classes that are symmetric along the y-axis.
            Defaults to None, which means no symmetric classes.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """
    def __init__(self, symmetric_classes: Optional[list[int]] = None, weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.symmetric_classes = symmetric_classes if symmetric_classes is not None else []

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.
        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``rotations``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``rotations``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_rotations = pred_instances.rotations
        gt_rotations = gt_instances.T[:, :3, :3]

        rot_dim = pred_rotations.shape[1]


        # create the transformation matrix
        if rot_dim == 6:
            r1, r2 = torch.split(pred_rotations, 3, dim=1)
            r1 = r1 / torch.norm(r1, dim=1, keepdim=True).clamp_min(1e-6)
            r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
            r2 = r2 / torch.norm(r2, dim=1, keepdim=True).clamp_min(1e-6)
            r3 = torch.cross(r1, r2, dim=1)
            pred_rotations = torch.stack([r1, r2, r3], dim=-1)
        elif rot_dim == 9:
            m = pred_rotations.view(-1, 3, 3)
            u, s, v = torch.svd(m)
            vt = torch.transpose(v, 1, 2)
            det = torch.det(torch.matmul(u, vt))
            det = det.view(-1, 1, 1)
            vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
            pred_rotations = torch.matmul(u, vt)
        else:
            raise ValueError(f'Unsupported rotation dimension: {self.rot_dim}')
        gt_labels = gt_instances.labels

        num_preds = pred_rotations.shape[0]
        num_gts = gt_rotations.shape[0]

        # repeat for batch-wise matrix multiplication
        pred_rotations = pred_rotations.unsqueeze(1).repeat(1, num_gts, 1, 1)
        gt_rotations = gt_rotations.unsqueeze(0).repeat(num_preds, 1, 1, 1)

        # Calculate the angle difference
        # batch-wise matrix multiplication: (num_preds, num_gts, 3, 3)
        rotation_diff_mat = pred_rotations @ gt_rotations.transpose(-1, -2)
        # trace: (num_preds, num_gts)
        cos_angle = (torch.einsum('...ii->...', rotation_diff_mat) - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        rotation_diff = torch.acos(cos_angle)

        # Handle symmetric classes
        if self.symmetric_classes:
            symmetric_mask = torch.zeros((num_gts), dtype=torch.bool, device=gt_labels.device)
            for sym_class in self.symmetric_classes:
                symmetric_mask = symmetric_mask | (gt_labels == sym_class)
            
            if torch.any(symmetric_mask):
                # rotation matrix for 180 degree rotation around y-axis
                R_y_pi = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                                      dtype=pred_rotations.dtype,
                                      device=pred_rotations.device)
                
                # (num_preds, num_gts, 3, 3)
                pred_rotations_sym = pred_rotations @ R_y_pi
                
                rotation_diff_mat_sym = pred_rotations_sym @ gt_rotations.transpose(-1, -2)
                cos_angle_sym = (torch.einsum('...ii->...', rotation_diff_mat_sym) - 1) / 2
                cos_angle_sym = torch.clamp(cos_angle_sym, -1.0, 1.0)
                rotation_diff_sym = torch.acos(cos_angle_sym)

                # (num_gts) -> (1, num_gts)
                sym_mask_expanded = symmetric_mask.unsqueeze(0)
                rotation_diff = torch.where(sym_mask_expanded, torch.min(rotation_diff, rotation_diff_sym), rotation_diff)

        return rotation_diff * self.weight

@TASK_UTILS.register_module()
class ADD9DCost(BaseMatchCost):
    """ADD cost. This assumes pred_instances contain rotations, translations, 
    sizes and gt_instances contain RTs and sizes.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
        symmetric_classes (list[int], optional): List of symmetric class ids.
            Defaults to None.
    """

    def __init__(self,
                 weight: Union[float, int] = 1.,
                 symmetric_classes: Optional[list[int]] = None) -> None:
        super().__init__(weight=weight)
        self.symmetric_classes = symmetric_classes if symmetric_classes is not None else []

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``rotations``, ``translations``,
                and ``sizes``.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It includes ``RTs`` and ``sizes``.
            img_meta (dict, optional): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_rotations = pred_instances.rotations
        pred_translations = pred_instances.translations
        pred_sizes = pred_instances.sizes

        gt_RTs = gt_instances.T
        gt_rotations = gt_RTs[:, :3, :3]
        gt_translations = gt_RTs[:, :3, 3]
        gt_sizes = gt_instances.sizes
        gt_labels = gt_instances.labels

        # Convert pred_rotations to rotation matrices
        r1, r2 = torch.split(pred_rotations, 3, dim=-1)
        r1 = r1 / torch.norm(r1, dim=-1, keepdim=True).clamp_min(1e-6)
        r2 = r2 - torch.bmm(r1.unsqueeze(1),
            r2.unsqueeze(-1)).squeeze(-1) * r1
        r2 = r2 / torch.norm(r2, dim=-1, keepdim=True).clamp_min(1e-6)
        r3 = torch.cross(r1, r2, dim=-1)
        pred_rotations = torch.stack((r1, r2, r3), dim=-1)

        # Define 8 corners of a cuboid
        unit_cube = torch.tensor([[-0.5, -0.5, 0],
                                  [0.5, -0.5, 0],
                                  [-0.5, 0.5, 0],
                                  [0.5, 0.5, 0],
                                  [-0.5, -0.5, 1],
                                  [0.5, -0.5, 1],
                                  [-0.5, 0.5, 1],
                                  [0.5, 0.5, 1]],
                                  dtype=torch.float32).to(pred_rotations.device)
        
        # Scale cuboid corners with sizes
        # pred_sizes: (num_preds, 3) -> (num_preds, 1, 3)
        # unit_cube: (8, 3)
        # pred_corners: (num_preds, 8, 3)
        pred_corners = pred_sizes.unsqueeze(1) * unit_cube
        gt_corners = gt_sizes.unsqueeze(1) * unit_cube

        # Transform corners
        # pred_rotations: (num_preds, 3, 3)
        # pred_corners: (num_preds, 8, 3) -> (num_preds, 3, 8)
        # pred_translations: (num_preds, 3) -> (num_preds, 3, 1)
        # pred_points: (num_preds, 3, 8) -> (num_preds, 8, 3)
        pred_points = (pred_rotations @ pred_corners.transpose(1, 2)) + pred_translations.unsqueeze(-1)
        pred_points = pred_points.transpose(1, 2)

        gt_points = (gt_rotations @ gt_corners.transpose(1, 2)) + gt_translations.unsqueeze(-1)
        gt_points = gt_points.transpose(1, 2)

        # Compute ADD cost
        # pred_points: (num_preds, 8, 3) -> (num_preds, 1, 8, 3)
        # gt_points: (num_gts, 8, 3) -> (1, num_gts, 8, 3)
        # cost: (num_preds, num_gts, 8, 3) -> (num_preds, num_gts)
        cost = torch.norm(pred_points.unsqueeze(1) - gt_points.unsqueeze(0), dim=-1).mean(dim=-1)

        # Handle symmetric classes
        if self.symmetric_classes:
            num_gts = len(gt_instances)
            symmetric_mask = torch.zeros((num_gts), dtype=torch.bool, device=gt_labels.device)
            for sym_class in self.symmetric_classes:
                symmetric_mask = symmetric_mask | (gt_labels == sym_class)

            if torch.any(symmetric_mask):
                # rotation matrix for 180 degree rotation around y-axis
                R_y_pi = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                                      dtype=pred_rotations.dtype,
                                      device=pred_rotations.device)
                
                pred_rotations_sym = pred_rotations @ R_y_pi
                pred_points_sym = (pred_rotations_sym @ pred_corners.transpose(1, 2)) + pred_translations.unsqueeze(-1)
                pred_points_sym = pred_points_sym.transpose(1, 2)
                
                cost_sym = torch.norm(pred_points_sym.unsqueeze(1) - gt_points.unsqueeze(0), dim=-1).mean(dim=-1)
                
                sym_mask_expanded = symmetric_mask.unsqueeze(0)
                cost = torch.where(sym_mask_expanded, torch.min(cost, cost_sym), cost)

        return cost * self.weight


def box2multiple_corners(box, num_points):
    B = box.size()[0]
    x, y, w, h, alpha = box.split([1, 1, 1, 1, 1], dim=-1)
    num_segments = num_points // 4

    weights = torch.linspace(0, 1, num_segments + 1,
                             device=box.device)[:-1]

    x_coords = torch.cat([
        -w / 2 + w * weights, w / 2 * torch.ones_like(weights),
        w / 2 - w * weights, -w / 2 * torch.ones_like(weights)
    ],
                         dim=-1)
    y_coords = torch.cat([
        h / 2 * torch.ones_like(weights), h / 2 - h * weights,
        -h / 2 * torch.ones_like(weights), -h / 2 + h * weights
    ],
                         dim=-1)
    corners = torch.stack([x_coords, y_coords], dim=-1)

    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)
    rot_T = torch.stack([row1, row2], dim=-2)

    N = box.size(1)
    rotated = torch.bmm(
        corners.view([-1, num_points, 2]),
        rot_T.repeat(1, 1, 1, 1).view([-1, 2, 2]))
    rotated = rotated.view([B, N, num_points, 2])
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def kld_loss_for_cost(pred, target, fun='log1p', tau=1.0, alpha=1.,
                      sqrt=True):
    """KLD distance used by GDCost (gaussian based match cost)."""
    xy_1, Sigma_1 = pred
    xy_2, Sigma_2 = target
    N, _ = xy_1.shape
    M, _ = xy_2.shape
    xy_1 = xy_1.unsqueeze(1).repeat(1, M, 1).view(-1, 2)
    Sigma_1 = Sigma_1.unsqueeze(1).repeat(1, M, 1, 1).view(-1, 2, 2)
    xy_2 = xy_2.unsqueeze(0).repeat(N, 1, 1).view(-1, 2)
    Sigma_2 = Sigma_2.unsqueeze(0).repeat(N, 1, 1, 1).view(-1, 2, 2)
    return_shape = [N, M]

    Sigma_1_inv = torch.stack(
        (Sigma_1[..., 1, 1], -Sigma_1[..., 0, 1],
         -Sigma_1[..., 1, 0], Sigma_1[..., 0, 0]),
        dim=-1).reshape(-1, 2, 2)
    Sigma_1_inv = Sigma_1_inv / Sigma_1.det().unsqueeze(-1).unsqueeze(-1)
    dxy = (xy_1 - xy_2).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2,
                                    1).bmm(Sigma_1_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_1_inv.bmm(Sigma_2).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)
    Sigma_1_det_log = Sigma_1.det().log()
    Sigma_2_det_log = Sigma_2.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_1_det_log -
                                         Sigma_2_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)

    if sqrt:
        distance = distance.clamp(1e-7).sqrt()
    distance = distance.reshape(return_shape)
    return postprocess(distance, fun=fun, tau=tau)


@TASK_UTILS.register_module()
class RBoxL1Cost(BBoxL1Cost):
    """Rotated bbox L1 match cost (5-dim: cx, cy, w, h, rad)."""

    def __init__(self,
                 box_format: str = 'xywha',
                 angle_factor=math.pi,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format == 'xywha'
        self.box_format = box_format
        self.angle_factor = angle_factor

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor(
            [img_w, img_h, img_w, img_h,
             self.angle_factor]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class CenterL1Cost(RBoxL1Cost):
    """Center L1 match cost (cx, cy only)."""

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        pred_bboxes = pred_bboxes[..., :2]
        gt_bboxes = gt_bboxes[..., :2]

        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class GDCost(BaseMatchCost):
    """Gaussian distance (KLD) match cost for rotated bboxes."""

    BAG_GD_COST = {
        'gwd': None,
        'kld': kld_loss_for_cost,
        'jd': None,
        'kld_symmax': None,
        'kld_symmin': None,
    }
    BAG_PREP = {
        'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
        'xy_wh_r': xy_wh_r_2_xy_sigma,
    }

    def __init__(self,
                 loss_type: str = 'kld',
                 representation: str = 'xy_wh_r',
                 fun: str = 'log1p',
                 tau: float = 0.0,
                 alpha: float = 1.0,
                 sqrt: bool = True,
                 weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        assert loss_type in self.BAG_GD_COST
        assert representation in self.BAG_PREP
        self.loss = self.BAG_GD_COST[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.sqrt = sqrt

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        # GDCost shares GDLoss's determinant/LU operation. It is used only by
        # Hungarian matching, so an fp32 calculation is both safe and required
        # when the detector forward uses fp16 autocast.
        with torch.autocast(device_type=pred_bboxes.device.type, enabled=False):
            pred_bboxes = self.preprocess(pred_bboxes.float())
            gt_bboxes = self.preprocess(gt_bboxes.float())
            gd_cost = self.loss(pred_bboxes, gt_bboxes, fun=self.fun,
                                tau=self.tau, alpha=self.alpha,
                                sqrt=self.sqrt)
            return gd_cost * self.weight


@TASK_UTILS.register_module()
class RotatedIoUCost(BaseMatchCost):
    """Rotated IoU match cost for rotated bboxes."""

    def __init__(self,
                 iou_mode: str = 'iou',
                 weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        if isinstance(pred_bboxes, BaseBoxes):
            pred_bboxes = pred_bboxes.tensor

        overlaps = rbbox_overlaps(
            pred_bboxes.to(gt_bboxes.device),
            gt_bboxes,
            mode=self.iou_mode)
        return (-overlaps) * self.weight
