from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco_panoptic_separated import dataloader
from ..common.models.panoptic_fpn import model
from ..common.train import train

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import RegNet
from detectron2.modeling.backbone.regnet import SimpleStem, ResBottleneckBlock
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

# train from scratch
train.init_checkpoint = ""
train.amp.enabled = True
train.ddp.fp16_compression = True
# RegNets benefit from enabling cudnn benchmark mode
train.cudnn_benchmark = True

# train.output_dir = 'output/panoptic_regnetx_0.4g'

model.backbone.bottom_up.freeze_at = 0
# model.backbone.bottom_up.freeze_at = 2
model.backbone.bottom_up = L(RegNet)(
    stem_class=SimpleStem,
    stem_width=32,
    block_class=ResBottleneckBlock,
    depth=22,
    w_a=24.48,
    w_0=24,
    w_m=2.54,
    group_width=16,
    norm="SyncBN",
    out_features=["s1", "s2", "s3", "s4"],
)
model.pixel_std = [57.375, 57.120, 58.395]
# model.roi_heads.box_predictor.test_score_thresh = 0.4


# image_size = 1024
# dataloader.train.mapper.augmentations = [
#     L(T.ResizeScale)(
#         min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
#     ),
#     L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
#     L(T.RandomFlip)(horizontal=True),
# ]
# # recompute boxes due to cropping
# dataloader.train.mapper.recompute_boxes = True
# larger batch-size.
dataloader.train.total_batch_size = 40
dataloader.test.num_workers = 1

# Equivalent to 100 epochs.
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[0.5, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.067,
)

optimizer.lr = 0.1
optimizer.weight_decay = 4e-5