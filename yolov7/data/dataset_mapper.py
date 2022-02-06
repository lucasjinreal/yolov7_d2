from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import transforms as T
from collections import deque
import copy
import logging
from typing import Optional, List, Union

import numpy as np
import torch
import cv2
import random

from detectron2.config import configurable, CfgNode
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import BoxMode

from .transforms.data_augment import random_perspective, box_candidates
from .detection_utils import build_augmentation, transform_instance_annotations, vis_annos
from yolov7.utils.boxes import adjust_box_anns


class MyDatasetMapper(DatasetMapper):

    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by YOLOF.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Add a queue for saving previous image infos in mosaic transformation
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(self,
                 is_train: bool,
                 *,
                 augmentations: List[Union[T.Augmentation, T.Transform]],
                 image_format: str,
                 mosaic_trans: Optional[CfgNode],
                 use_instance_mask: bool = False,
                 use_keypoint: bool = False,
                 instance_mask_format: str = "polygon",
                 recompute_boxes: bool = False,
                 add_meta_infos: bool = False):
        """
        Args:
            augmentations: a list of augmentations or deterministic
                transforms to apply
            image_format: an image format supported by
                :func:`detection_utils.read_image`.
            mosaic_trans: a CfgNode for Mosaic transformation.
            use_instance_mask: whether to process instance segmentation
                annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process
                instance segmentation masks into this format.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask
                annotations.
            add_meta_infos: whether to add `meta_infos` field
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.recompute_boxes = recompute_boxes
        self.add_meta_infos = add_meta_infos
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.mosaic_trans = mosaic_trans
        if self.mosaic_trans.ENABLED:
            self.mosaic_pool = deque(
                maxlen=self.mosaic_trans.POOL_CAPACITY)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # use a local `build_augmentation` instead
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0,
                        T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        elif cfg.INPUT.JITTER_CROP.ENABLED and is_train:
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "mosaic_trans": cfg.INPUT.MOSAIC,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "add_meta_infos": cfg.INPUT.JITTER_CROP.ENABLED,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset
                format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # add image info to mosaic pool
        mosaic_flag = 0
        mosaic_samples = None
        if self.mosaic_trans.ENABLED and self.is_train:
            if len(self.mosaic_pool) > self.mosaic_trans.NUM_IMAGES:
                mosaic_flag = np.random.randint(2)
                # sample images in the mosaic_pool
                if mosaic_flag == 1:
                    mosaic_samples = np.random.choice(
                        self.mosaic_pool,
                        self.mosaic_trans.NUM_IMAGES - 1)
            self.mosaic_pool.append(copy.deepcopy(dataset_dict))

        # for current image
        image, annos = self._load_image_with_annos(dataset_dict)

        if self.is_train and mosaic_flag == 1 and mosaic_samples is not None:
            min_offset = self.mosaic_trans.MIN_OFFSET
            mosaic_width = self.mosaic_trans.MOSAIC_WIDTH
            mosaic_height = self.mosaic_trans.MOSAIC_HEIGHT
            cut_x = np.random.randint(int(mosaic_width * min_offset),
                                      int(mosaic_width * (1 - min_offset)))
            cut_y = np.random.randint(int(mosaic_height * min_offset),
                                      int(mosaic_height * (1 - min_offset)))
            # init the out image and the out annotations
            out_image = np.zeros(
                [mosaic_height, mosaic_width, 3], dtype=image.dtype)
            out_annos = []
            # mosaic transform
            for m_idx in range(self.mosaic_trans.NUM_IMAGES):
                # re-load the image and annotations for the sampled images
                # replace the current image and annos with the new image's
                if m_idx != 0:
                    dataset_dict = copy.deepcopy(mosaic_samples[m_idx - 1])
                    image, annos = self._load_image_with_annos(dataset_dict)

                image_size = image.shape[:2]  # h, w
                # as all meta_infos are the same, we just get the first one
                meta_infos = annos[0].pop("meta_infos")
                pleft = meta_infos.get('jitter_pad_left', 0)
                pright = meta_infos.get('jitter_pad_right', 0)
                ptop = meta_infos.get('jitter_pad_top', 0)
                pbot = meta_infos.get('jitter_pad_bot', 0)
                # get shifts
                left_shift = min(cut_x, max(0, -int(pleft)))
                top_shift = min(cut_y, max(0, -int(ptop)))
                right_shift = min(image_size[1] - cut_x, max(0, -int(pright)))
                bot_shift = min(image_size[0] - cut_y, max(0, -int(pbot)))
                out_image, cur_annos = self._blend_moasic(
                    cut_x,
                    cut_y,
                    out_image,
                    image,
                    copy.deepcopy(annos),
                    (mosaic_height, mosaic_width),
                    m_idx,
                    (left_shift, top_shift, right_shift, bot_shift)
                )
                out_annos.extend(cur_annos)
            # replace image and annotation with out_image and out_annotation
            image, annos = out_image, out_annos
            # print(image)
            # print(annos)
            if self.mosaic_trans.DEBUG_VIS:
                a = np.array(image).astype(np.uint8)
                a = vis_annos(a, annos)

        if annos is not None:
            image_shape = image.shape[:2]  # h, w
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box
            # may no longer tightly bound the object. As an example, imagine
            # a triangle object [(0,0), (2,0), (0,2)] cropped by a box [(1,
            # 0),(2,2)] (XYXY format). The tight bounding box of the cropped
            # triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # Pytorch's dataloader is efficient on torch.Tensor due to
        # shared-memory,
        # but not efficient on large generic data structures due to the use
        # of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        return dataset_dict

    def _load_image_with_annos(self, dataset_dict):
        """
        Load the image and annotations given a dataset_dict.
        """
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"],
                                 format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return image, None

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other
            # types of data
            # apply meta_infos for mosaic transformation
            annos = [
                transform_instance_annotations(
                    obj, transforms, image_shape,
                    add_meta_infos=self.add_meta_infos
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            annos = None
        return image, annos

    def _apply_boxes(self,
                     annotations,
                     left_shift,
                     top_shift,
                     cut_width,
                     cut_height,
                     cut_start_x,
                     cut_start_y):
        """
        Modify the boxes' coordinates according to shifts and cut_starts.
        """
        for annotation in annotations:
            bboxes = BoxMode.convert(annotation["bbox"],
                                     annotation["bbox_mode"],
                                     BoxMode.XYXY_ABS)
            bboxes = np.asarray(bboxes)
            bboxes[0::2] -= left_shift
            bboxes[1::2] -= top_shift

            bboxes[0::2] = np.clip(bboxes[0::2], 0, cut_width)
            bboxes[1::2] = np.clip(bboxes[1::2], 0, cut_height)
            bboxes[0::2] += cut_start_x
            bboxes[1::2] += cut_start_y
            annotation["bbox"] = bboxes
            annotation["bbox_mode"] = BoxMode.XYXY_ABS
        return annotations

    def _blend_moasic(self,
                      cut_x,
                      cut_y,
                      target_img,
                      img,
                      annos,
                      img_size,
                      blend_index,
                      four_shifts):
        """
        Blend the images and annotations in Mosaic transform.
        """
        h, w = img_size
        img_h, img_w = img.shape[:2]
        left_shift = min(four_shifts[0], img_w - cut_x)
        top_shift = min(four_shifts[1], img_h - cut_y)
        right_shift = min(four_shifts[2], img_w - (w - cut_x))
        bot_shift = min(four_shifts[3], img_h - (h - cut_y))

        if blend_index == 0:
            annos = self._apply_boxes(
                annos, left_shift, top_shift, cut_x, cut_y, 0, 0
            )
            target_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y,
                                             left_shift:left_shift + cut_x]
        if blend_index == 1:
            annos = self._apply_boxes(
                annos, img_w + cut_x - w - right_shift,
                top_shift, w - cut_x, cut_y, cut_x, 0
            )
            target_img[:cut_y, cut_x:] = \
                img[top_shift:top_shift + cut_y,
                    img_w + cut_x - w - right_shift:img_w - right_shift]
        if blend_index == 2:
            annos = self._apply_boxes(
                annos, left_shift, img_h + cut_y - h - bot_shift,
                cut_x, h - cut_y, 0, cut_y
            )
            target_img[cut_y:, :cut_x] = \
                img[img_h + cut_y - h - bot_shift:img_h - bot_shift,
                    left_shift:left_shift + cut_x]
        if blend_index == 3:
            annos = self._apply_boxes(annos, img_w + cut_x - w - right_shift,
                                      img_h + cut_y - h - bot_shift,
                                      w - cut_x, h - cut_y, cut_x, cut_y)
            target_img[cut_y:, cut_x:] = \
                img[img_h + cut_y - h - bot_shift:img_h - bot_shift,
                    img_w + cut_x - w - right_shift:img_w - right_shift]
        return target_img, annos


class MyDatasetMapper2(DatasetMapper):

    """
    Add another version mosiac here, also enables MIXUP

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Add a queue for saving previous image infos in mosaic transformation
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(self,
                 is_train: bool,
                 *,
                 augmentations: List[Union[T.Augmentation, T.Transform]],
                 image_format: str,
                 mosaic_trans: Optional[CfgNode],
                 use_instance_mask: bool = False,
                 use_keypoint: bool = False,
                 instance_mask_format: str = "polygon",
                 recompute_boxes: bool = False,
                 add_meta_infos: bool = False,
                 input_size: List = [640, 640]):
        """
        Args:
            augmentations: a list of augmentations or deterministic
                transforms to apply
            image_format: an image format supported by
                :func:`detection_utils.read_image`.
            mosaic_trans: a CfgNode for Mosaic transformation.
            use_instance_mask: whether to process instance segmentation
                annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process
                instance segmentation masks into this format.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask
                annotations.
            add_meta_infos: whether to add `meta_infos` field
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.recompute_boxes = recompute_boxes
        self.add_meta_infos = add_meta_infos
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.mosaic_trans = mosaic_trans
        self.enable_aug = True
        self.input_size = input_size
        if self.mosaic_trans.ENABLED:
            self.mosaic_pool = deque(
                maxlen=self.mosaic_trans.POOL_CAPACITY)

            self.degrees = self.mosaic_trans.DEGREES
            self.translate = self.mosaic_trans.TRANSLATE
            self.scale = self.mosaic_trans.SCALE
            self.mixup_scale = self.mosaic_trans.MSCALE
            self.shear = self.mosaic_trans.SHEAR
            self.perspective = self.mosaic_trans.PERSPECTIVE
            self.enable_mixup = self.mosaic_trans.ENABLE_MIXUP
    
    @classmethod
    def disable_aug(self):
        self.enable_aug = False

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # use a local `build_augmentation` instead
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0,
                        T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "mosaic_trans": cfg.INPUT.MOSAIC_AND_MIXUP,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "add_meta_infos": cfg.INPUT.JITTER_CROP.ENABLED,
            "input_size": cfg.INPUT.INPUT_SIZE
        }
        return ret

    def _anno_to_labels(self, annos):
        """
        annos to labels:

        [x1, y1, x2, y2, cls]
        """
        labels = []
        if annos is not None:
            for ann in annos:
                b = ann['bbox']
                c = ann['category_id']
                labels.append(np.append(b, c))
        return np.array(labels)

    def _labels_to_annos(self, labels):
        annos = []
        if labels.shape[0] == 0:
            # an empty box for: https://github.com/facebookresearch/detectron2/issues/3401
            r = dict()
            # this dummy box will be filtered in filter_empty box
            r['bbox'] = [0, 0, 0, 0]
            r['category_id'] = 0
            r['bbox_mode'] = BoxMode.XYXY_ABS
            return [r]
        for i in range(labels.shape[0]):
            b = labels[i, :][:4]
            c = labels[i, :][4]
            r = dict()
            r['bbox'] = b
            r['category_id'] = c
            r['bbox_mode'] = BoxMode.XYXY_ABS
            annos.append(r)
        return annos

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset
                format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # add image info to mosaic pool
        mosaic_flag = 0
        mosaic_samples = None
        if self.mosaic_trans.ENABLED and self.is_train and self.enable_aug:
            if len(self.mosaic_pool) > self.mosaic_trans.NUM_IMAGES:
                mosaic_flag = np.random.randint(2)
                # sample images in the mosaic_pool
                if mosaic_flag == 1:
                    mosaic_samples = np.random.choice(
                        self.mosaic_pool,
                        self.mosaic_trans.NUM_IMAGES - 1)
            self.mosaic_pool.append(copy.deepcopy(dataset_dict))
        
        # print(len(self.mosaic_pool), mosaic_samples)
        # for current image
        img, annos = self._load_image_with_annos(dataset_dict)

        if self.is_train and mosaic_flag == 1 and mosaic_samples is not None and self.enable_aug:
            # input_dim = self._dataset.input_dim
            # (todo): hard code to input_dim as 640,640
            # input_dim = (640, 640)
            w = np.random.randint(
                self.mosaic_trans.MOSAIC_WIDTH_RANGE[0], self.mosaic_trans.MOSAIC_WIDTH_RANGE[1]+1)
            h = np.random.randint(
                self.mosaic_trans.MOSAIC_HEIGHT_RANGE[0], self.mosaic_trans.MOSAIC_HEIGHT_RANGE[1]+1)
            if max(w/h, h/w) > 1.2:
                # ratio abnormal
                h = min(h, w)
                w = int(1.2*h)
            input_dim = (h, w)  # h, w
            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_dim[0], 1.5 * input_dim[0]))
            xc = int(random.uniform(0.5 * input_dim[1], 1.5 * input_dim[1]))

            out_annos = []
            labels4 = []
            for i in range(self.mosaic_trans.NUM_IMAGES):
                if i != 0:
                    dataset_dict = copy.deepcopy(mosaic_samples[i - 1])
                    img, annos = self._load_image_with_annos(dataset_dict)
                    # print(i, annos)

                image_size = img.shape[: 2]
                _labels = self._anno_to_labels(annos)

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_dim[0] / h0, 1. * input_dim[1] / w0)
                interp = cv2.INTER_LINEAR
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=interp)
                (h, w) = img.shape[:2]

                if i == 0:  # top left
                    # base image with 4 tiles
                    img4 = np.full(
                        (input_dim[0] * 2, input_dim[1] * 2, img.shape[2]), 114, dtype=np.uint8
                    )
                    # xmin, ymin, xmax, ymax (large image)
                    x1a, y1a, x2a, y2a = (
                        max(xc - w, 0), max(yc - h, 0), xc, yc,)
                    # xmin, ymin, xmax, ymax (small image)
                    x1b, y1b, x2b, y2b = (
                        w - (x2a - x1a), h - (y2a - y1a), w, h,)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(
                        yc - h, 0), min(xc + w, input_dim[1] * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - \
                        (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(
                        xc - w, 0), yc, xc, min(input_dim[0] * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - \
                        (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, input_dim[1] * 2), min(input_dim[0] * 2, yc + h)  # noqa
                    x1b, y1b, x2b, y2b = 0, 0, min(
                        w, x2a - x1a), min(y2a - y1a, h)

                # img4[ymin:ymax, xmin:xmax]
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b

                # [[xmin, ymin, xmax, ymax, label_ind], ... ]
                labels = _labels.copy()
                if _labels.size > 0:  # Normalized xywh to pixel xyxy format
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh

                    # if labels empty skip it.
                    labels4.append(labels)

            if len(labels4):
                labels4 = np.concatenate(labels4, 0)
                np.clip(labels4[:, 0], 0, 2 * input_dim[1], out=labels4[:, 0])
                np.clip(labels4[:, 1], 0, 2 * input_dim[0], out=labels4[:, 1])
                np.clip(labels4[:, 2], 0, 2 * input_dim[1], out=labels4[:, 2])
                np.clip(labels4[:, 3], 0, 2 * input_dim[0], out=labels4[:, 3])

            img4, labels4 = random_perspective(
                img4,
                labels4,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_dim[0] // 2, -input_dim[1] // 2],
            )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if self.enable_mixup and not len(labels4) == 0:
                img4, labels4 = self.mixup(img4, labels4, input_dim)
            # mix_img, padded_labels = self.preproc(
            #     img4, labels4, input_dim)
            # img_info = (img4.shape[1], mix_img.shape[0])

            out_annos = self._labels_to_annos(labels4)
            # make labels back to annos

            img, annos = img4, out_annos
            if self.mosaic_trans.DEBUG_VIS:
                a = np.array(img).astype(np.uint8)
                a = vis_annos(a, annos)

        if annos is not None:
            image_shape = img.shape[:2]  # h, w
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box
            # may no longer tightly bound the object. As an example, imagine
            # a triangle object [(0,0), (2,0), (0,2)] cropped by a box [(1,
            # 0),(2,2)] (XYXY format). The tight bounding box of the cropped
            # triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # Pytorch's dataloader is efficient on torch.Tensor due to
        # shared-memory,
        # but not efficient on large generic data structures due to the use
        # of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1)))
        return dataset_dict

    def _load_image_with_annos(self, dataset_dict):
        """
        Load the image and annotations given a dataset_dict.
        """
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"],
                                 format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return image, None

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other
            # types of data
            # apply meta_infos for mosaic transformation
            annos = [
                transform_instance_annotations(
                    obj, transforms, image_shape,
                    add_meta_infos=self.add_meta_infos
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            print('no annotations in data dict')
            annos = None
        return image, annos

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        # cp_labels = []
        # while len(cp_labels) == 0:
            # cp_index = random.randint(0, self.__len__() - 1)
            # cp_labels = self._dataset.load_anno(cp_index)

        annos = None
        while annos == None:
            # must found a sample with annotation
            cp_samples = copy.deepcopy(np.random.choice(self.mosaic_pool, 1)[0])
            img, annos = self._load_image_with_annos(cp_samples)
        cp_labels = self._anno_to_labels(annos)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0
        cp_scale_ratio = min(
            input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio),
             int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor),
             int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4], cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(
            cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5]
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * \
                padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels



"""

DatasetMapper for detr
"""


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict