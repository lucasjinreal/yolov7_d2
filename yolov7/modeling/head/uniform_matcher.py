import numpy as np
import torch
from torch import nn


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class UniformMatcher(nn.Module):
    """
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Note:
        In this implementation, we combine the indexes of topk anchors and
        topk predicted boxes. There exist duplicates between the indexes of
        topk predicts boxes and the indexes of topk anchors. The topk
        predict boxes' non-duplicate indexes serve as additional indexes,
        which help the model train better. This implementation gives stable
        and slightly higher performance than only using topk anchors.


    Args:
        match_times(int): Number of positive anchors for each gt box.
    """

    def __init__(self, match_times: int = 4):
        super().__init__()
        self.match_times = match_times

    @torch.no_grad()
    def forward(self, pred_boxes, anchors, targets):
        bs, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_anchors, 4]
        out_bbox = pred_boxes.flatten(0, 1)
        anchors = anchors.flatten(0, 1)

        # Also concat the target boxes
        tgt_bbox = torch.cat([v.gt_boxes.tensor for v in targets])

        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(tgt_bbox), p=1)
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(tgt_bbox), p=1)

        # Final cost matrix
        C = cost_bbox
        C = C.view(bs, num_queries, -1).cpu()
        C1 = cost_bbox_anchors
        C1 = C1.view(bs, num_queries, -1).cpu()

        sizes = [len(v.gt_boxes.tensor) for v in targets]
        all_indices_list = [[] for _ in range(bs)]
        # positive indices when matching predict boxes and gt boxes
        indices = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist()
            )
            for i, c in enumerate(C.split(sizes, -1))
        ]
        # positive indices when matching anchor boxes and gt boxes
        indices1 = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist())
            for i, c in enumerate(C1.split(sizes, -1))]

        # concat the indices according to image ids
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):
            img_idx_i = [
                np.array(idx_ + idx1_)
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            img_idx_j = [
                np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)]

        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))
        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in all_indices
        ]
