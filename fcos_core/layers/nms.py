# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import torch

def ml_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, max_output_boxes=200):
    scores = scores.clone()
    keep = []

    while scores.numel() > 0:
        max_score_idx = scores.argmax()
        max_score_box = boxes[max_score_idx].unsqueeze(0)

        keep.append(max_score_idx.item())

        if len(scores) == 1:
            break

        ious = box_iou(max_score_box, boxes)
        weights = torch.exp(-(ious ** 2) / sigma)
        scores = scores * weights.squeeze()

        if len(scores) > max_output_boxes:
            scores = scores[:max_output_boxes]
            boxes = boxes[:max_output_boxes]

    return torch.tensor(keep, dtype=torch.long)

def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter = (
        (torch.min(box1[:, None, 2:], box2[:, 2:]) -
         torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    )
    union = area1[:, None] + area2 - inter
    return inter / union
def nms(boxes, scores, iou_threshold):
    keep = []
    indices = scores.argsort(descending=True)

    while indices.numel() > 0:
        i = indices[0].item()
        keep.append(i)
        if indices.numel() == 1:
            break

        ious = box_iou(boxes[i, :].unsqueeze(0), boxes[indices[1:], :]).squeeze()
        indices = indices[1:][ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)

nms = nms
ml_nms = ml_nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""