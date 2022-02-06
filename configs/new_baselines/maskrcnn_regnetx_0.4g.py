from .mask_rcnn_R_50_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    optimizer,
    train,
)
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import RegNet
from detectron2.modeling.backbone.regnet import SimpleStem, ResBottleneckBlock

from ..common.models.mask_rcnn_fpn import model

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



train.max_iter *= 2  # 100ep -> 200ep

lr_multiplier.scheduler.milestones = [
    milestone * 2 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter
