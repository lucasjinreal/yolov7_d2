from detectron2.layers.nms import batched_nms
import torch
from torchvision.ops import boxes as box_ops


def iou(boxes, top_box):
    x1 = boxes[:, 0].clamp(min=top_box[0])
    y1 = boxes[:, 1].clamp(min=top_box[1])
    x2 = boxes[:, 2].clamp(max=top_box[2])
    y2 = boxes[:, 3].clamp(max=top_box[3])

    inters = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unions = (top_box[2] - top_box[0]) * \
        (top_box[3] - top_box[1]) + areas - inters

    return inters / unions


def scale_by_iou(ious, sigma, soft_mode="gaussian"):
    if soft_mode == "linear":
        scale = ious.new_ones(ious.size())
        scale[ious >= sigma] = 1 - ious[ious >= sigma]
    else:
        scale = torch.exp(-ious ** 2 / sigma)

    return scale


def softnms(boxes, scores, sigma, score_threshold, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]

    undone_mask = scores >= score_threshold
    while undone_mask.sum() > 1:
        idx = scores[undone_mask].argmax()
        idx = undone_mask.nonzero(as_tuple=False)[idx].item()
        top_box = boxes[idx]
        undone_mask[idx] = False
        _boxes = boxes[undone_mask]

        ious = iou(_boxes, top_box)
        scales = scale_by_iou(ious, sigma, soft_mode)

        scores[undone_mask] *= scales
        undone_mask[scores < score_threshold] = False
    return scores


def batched_softnms(boxes, scores, idxs, iou_threshold,
                    score_threshold=0.001, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]
    assert boxes.shape[-1] == 4

    # change scores inplace
    # no need to return changed scores
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        scores[mask] = softnms(boxes[mask], scores[mask], iou_threshold,
                               score_threshold, soft_mode)

    keep = (scores > score_threshold).nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def cluster_nms(boxes, scores, iou_threshold):
    last_keep = torch.ones(*scores.shape).to(boxes.device)

    scores, idx = scores.sort(descending=True)
    boxes = boxes[idx]
    origin_iou_matrix = box_ops.box_iou(
        boxes, boxes).tril(diagonal=-1).transpose(1, 0)

    while True:
        iou_matrix = torch.mm(torch.diag(last_keep.float()), origin_iou_matrix)
        keep = (iou_matrix.max(dim=0)[0] <= iou_threshold)

        if (keep == last_keep).all():
            return idx[keep.nonzero(as_tuple=False)]

        last_keep = keep


def batched_clusternms(boxes, scores, idxs, iou_threshold):
    assert boxes.shape[-1] == 4

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = cluster_nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def generalized_batched_nms(boxes, scores, idxs, iou_threshold,
                            score_threshold=0.001, nms_type="normal"):
    assert boxes.shape[-1] == 4

    if nms_type == "normal":
        keep = batched_nms(boxes, scores, idxs, iou_threshold)
    elif nms_type.startswith("softnms"):
        keep = batched_softnms(boxes, scores, idxs, iou_threshold,
                               score_threshold=score_threshold,
                               soft_mode=nms_type.lstrip("softnms-"))
    elif nms_type == "cluster":
        keep = batched_clusternms(boxes, scores, idxs, iou_threshold)
    else:
        raise NotImplementedError(
            "NMS type not implemented: \"{}\"".format(nms_type))

    return keep
