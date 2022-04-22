# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

SPARSE_INST_DECODER_REGISTRY = Registry("SPARSE_INST_DECODER")
SPARSE_INST_DECODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


class InstanceBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        # norm = cfg.MODEL.SPARSE_INST.DECODER.NORM
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        inst_features = inst_features / normalizer[:, :, None]
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


class MaskBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


@SPARSE_INST_DECODER_REGISTRY.register()
class BaseIAMDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # add 2 for coordinates
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2

        self.scale_factor = cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR
        self.output_iam = cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM

        self.inst_branch = InstanceBranch(cfg, in_channels)
        self.mask_branch = MaskBranch(cfg, in_channels)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
        mask_features = self.mask_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        if torch.onnx.is_in_onnx_export():
            sh = mask_features.shape
            pred_masks = torch.bmm(pred_kernel, mask_features.view(sh[0], sh[1], sh[2] * sh[3])).view(sh[0], N, sh[2], sh[3])
        else:
            B, C, H, W = mask_features.shape
            pred_masks = torch.bmm(pred_kernel, mask_features.view(B, C, H * W)).view(
                B, N, H, W
            )

        pred_masks = F.interpolate(
            pred_masks,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )

        output = {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
        }

        if self.output_iam:
            iam = F.interpolate(
                iam,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
            output["pred_iam"] = iam

        return output


class GroupInstanceBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.num_groups = cfg.MODEL.SPARSE_INST.DECODER.GROUPS
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a group conv
        expand_dim = dim * self.num_groups
        self.iam_conv = nn.Conv2d(
            dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups
        )
        # outputs
        self.fc = nn.Linear(expand_dim, expand_dim)

        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        normalizer = iam_prob.sum(-1).clamp(min=1e-6, max=1e5)
        # print(normalizer)
        inst_features = inst_features / normalizer[:, :, None]

        # d4 = torch.div(N, 4, rounding_mode='floor') # can't use this for onnx tracable
        d4 = N // 4
        inst_features = (
            inst_features.reshape(B, 4, d4, -1)
            .transpose(1, 2)
            .reshape(B, d4, -1)
        )

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


@SPARSE_INST_DECODER_REGISTRY.register()
class GroupIAMDecoder(BaseIAMDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2
        self.inst_branch = GroupInstanceBranch(cfg, in_channels)


def build_sparse_inst_decoder(cfg):
    name = cfg.MODEL.SPARSE_INST.DECODER.NAME
    return SPARSE_INST_DECODER_REGISTRY.get(name)(cfg)
