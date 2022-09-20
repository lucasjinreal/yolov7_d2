from detectron2.data import transforms as T
from detectron2.structures import BoxMode
import numpy as np
from alfred.utils.log import logger
from .transforms.augmentation_impl import (
    YOLOFJitterCrop,
    YOLOFResize,
    YOLOFRandomDistortion,
    # RandomFlip,
    YOLOFRandomShift,
    RandomGridMask
)
from detectron2.data.transforms import RandomFlip, RandomBrightness, RandomLighting, RandomSaturation
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy
from pycocotools import mask as mask_util


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    is_normal_aug = not cfg.INPUT.RESIZE.ENABLED
    if is_normal_aug:
        augmentation = build_normal_augmentation(cfg, is_train)
    else:
        augmentation = build_yolov7_augmentation(cfg, is_train)
    if is_train and cfg.INPUT.SHIFT.ENABLED and cfg.INPUT.SHIFT.SHIFT_PIXELS>0:
        augmentation.append(
            YOLOFRandomShift(max_shifts=cfg.INPUT.SHIFT.SHIFT_PIXELS))
    return augmentation


def build_normal_augmentation(cfg, is_train):
    """
    Train Augmentations:
        - ResizeShortestEdge
        - RandomFlip (not for test)
    Test:
        - ResizeShortestEdge
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP_HORIZONTAL.ENABLED:
        augmentation.append(
            T.RandomFlip(
                prob=cfg.INPUT.RANDOM_FLIP_HORIZONTAL.PROB,
                horizontal=cfg.INPUT.RANDOM_FLIP_HORIZONTAL.ENABLED,
                vertical=False,
            )
        )
    if is_train and cfg.INPUT.RANDOM_FLIP_VERTICAL.ENABLED:
        augmentation.append(
            T.RandomFlip(
                prob=cfg.INPUT.RANDOM_FLIP_VERTICAL.PROB,
                horizontal=False,
                vertical=cfg.INPUT.RANDOM_FLIP_VERTICAL.ENABLED,
            )
        )
    if is_train and cfg.INPUT.COLOR_JITTER.SATURATION:
        augmentation.append(RandomSaturation(0.8, 1.2))
    if is_train and cfg.INPUT.COLOR_JITTER.BRIGHTNESS:
        augmentation.append(RandomBrightness(0.8, 1.2))

    if is_train and cfg.INPUT.DISTORTION.ENABLED:
        augmentation.append(
            YOLOFRandomDistortion(
                hue=cfg.INPUT.DISTORTION.HUE,
                saturation=cfg.INPUT.DISTORTION.SATURATION,
                exposure=cfg.INPUT.DISTORTION.EXPOSURE
            )
        )
    if is_train and cfg.INPUT.GRID_MASK.ENABLED:
        augmentation.append(
            RandomGridMask(prob=cfg.INPUT.GRID_MASK.PROB,
                           use_h=cfg.INPUT.GRID_MASK.USE_HEIGHT, use_w=cfg.INPUT.GRID_MASK.USE_WIDTH, mode=cfg.INPUT.GRID_MASK.MODE))
    return augmentation


def build_yolov7_augmentation(cfg, is_train):
    augmentation = []
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        augmentation.append(T.ResizeShortestEdge(
            min_size, max_size, sample_style))

        if cfg.INPUT.JITTER_CROP.ENABLED:
            augmentation.append(YOLOFJitterCrop(
                cfg.INPUT.JITTER_CROP.JITTER_RATIO))
        if cfg.INPUT.MOSAIC.ENABLED:
            augmentation.append(
                YOLOFResize(shape=cfg.INPUT.RESIZE.SHAPE,
                            scale_jitter=cfg.INPUT.RESIZE.SCALE_JITTER)
            )
        if cfg.INPUT.DISTORTION.ENABLED:
            augmentation.append(
                YOLOFRandomDistortion(
                    hue=cfg.INPUT.DISTORTION.HUE,
                    saturation=cfg.INPUT.DISTORTION.SATURATION,
                    exposure=cfg.INPUT.DISTORTION.EXPOSURE
                )
            )
        if cfg.INPUT.GRID_MASK.ENABLED:
            augmentation.append(
                RandomGridMask(prob=cfg.INPUT.GRID_MASK.PROB,
                               use_h=cfg.INPUT.GRID_MASK.USE_HEIGHT, use_w=cfg.INPUT.GRID_MASK.USE_WIDTH, mode=cfg.INPUT.GRID_MASK.MODE)
            )
        if cfg.INPUT.COLOR_JITTER.SATURATION:
            augmentation.append(RandomSaturation(0.8, 1.2))
        if cfg.INPUT.COLOR_JITTER.BRIGHTNESS:
            augmentation.append(RandomBrightness(0.8, 1.2))
        
        # The difference between `T.RandomFlip` and `RandomFlip` is that
        # we register a new method `apply_meta_infos` in `RandomFlip`
        if cfg.INPUT.RANDOM_FLIP_HORIZONTAL.ENABLED:
            augmentation.append(
                T.RandomFlip(
                    prob=cfg.INPUT.RANDOM_FLIP_HORIZONTAL.PROB,
                    horizontal=cfg.INPUT.RANDOM_FLIP_HORIZONTAL.ENABLED,
                    vertical=False,
                )
            )
        if cfg.INPUT.RANDOM_FLIP_VERTICAL.ENABLED:
            augmentation.append(
                T.RandomFlip(
                    prob=cfg.INPUT.RANDOM_FLIP_VERTICAL.PROB,
                    horizontal=False,
                    vertical=cfg.INPUT.RANDOM_FLIP_VERTICAL.ENABLED,
                )
            )

    else:
        # augmentation.append(
        #     YOLOFResize(shape=cfg.INPUT.RESIZE.TEST_SHAPE,
        #                 scale_jitter=None)
        # )
        # we don't need forced resize, just keep ratio
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        augmentation.append(T.ResizeShortestEdge(
            min_size, max_size, sample_style))
    return augmentation


def transform_instance_annotations(
        annotation, transforms, image_size, *, add_meta_infos=False
):
    """
    Apply transforms to box and meta_infos annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        add_meta_infos (bool): Whether to apply meta_infos.

    Returns:
        dict:
            the same input dict with fields "bbox", "meta_infos"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    
    # apply transforms to segmentation
    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    # add meta_infos
    if add_meta_infos:
        meta_infos = dict()
        meta_infos = transforms.apply_meta_infos(meta_infos)
        annotation["meta_infos"] = meta_infos
    return annotation


def vis_annos(img, annos):
    bboxes = []
    clss = []
    scores = []
    for ann in annos:
        bboxes.append(ann['bbox'].tolist())
        clss.append(ann['category_id'])
        scores.append(1)
    bboxes = np.array(bboxes)

    img = visualize_det_cv2_part(
        img, None, clss, bboxes, is_show=True, line_thickness=1)
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img
