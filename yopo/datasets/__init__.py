# Copyright (c) OpenMMLab. All rights reserved.
from .base_det_dataset import BaseDetDataset
from .coco import CocoDataset
from .coco_caption import CocoCaptionDataset
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import ConcatDataset, MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .dod import DODDataset
from .dota_tomato import DOTAOBBDataset, DOTATomatoDataset
from .dsdl import DSDLDetDataset
from .flickr30k import Flickr30kDataset
from .isaid import iSAIDDataset
from .lvis import LVISDataset, LVISV05Dataset, LVISV1Dataset
from .mdetr_style_refcoco import MDETRStyleRefCocoDataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .odvg import ODVGDataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .pose_estimation import (
    HouseCat6DDataset,
    NOCSDataset,
    Wild6DDataset,
    YCBVideoBOPDataset,
    YOPOSequenceDataset,
)
from .refcoco import RefCocoDataset
from .samplers import (
    AspectRatioBatchSampler,
    ClassAwareSampler,
    CustomSampleSizeSampler,
    GroupMultiSourceSampler,
    MultiSourceSampler,
)
from .utils import get_loading_pipeline
from .v3det import V3DetDataset
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset

__all__ = [
    "XMLDataset",
    "CocoDataset",
    "DeepFashionDataset",
    "VOCDataset",
    "LVISDataset",
    "LVISV05Dataset",
    "LVISV1Dataset",
    "WIDERFaceDataset",
    "get_loading_pipeline",
    "MultiImageMixDataset",
    "OpenImagesDataset",
    "OpenImagesChallengeDataset",
    "AspectRatioBatchSampler",
    "ClassAwareSampler",
    "MultiSourceSampler",
    "GroupMultiSourceSampler",
    "BaseDetDataset",
    "CrowdHumanDataset",
    "Objects365V1Dataset",
    "Objects365V2Dataset",
    "DSDLDetDataset",
    "V3DetDataset",
    "ConcatDataset",
    "ODVGDataset",
    "MDETRStyleRefCocoDataset",
    "DODDataset",
    "CustomSampleSizeSampler",
    "Flickr30kDataset",
    "CocoCaptionDataset",
    "RefCocoDataset",
    "iSAIDDataset",
    "YCBVideoBOPDataset",
    "NOCSDataset",
    "HouseCat6DDataset",
    "Wild6DDataset",
    "DOTAOBBDataset",
    "DOTATomatoDataset",
    "YOPOSequenceDataset",
]
