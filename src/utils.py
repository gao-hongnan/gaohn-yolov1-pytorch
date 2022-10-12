import ast
import json
import os
import re
from shutil import copyfile

import numpy as np
import pandas as pd


from pathlib import Path
import torch
from typing import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import os
import random
from typing import Union
import copy

BboxType = Union[np.ndarray, torch.Tensor]

import numpy as np


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if isinstance(a, list):
        a = np.asarray(a)
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = (
        np.array2string(
            a,
            max_line_width=np.infty,
            # formatter={"float_kind": lambda x: "{:.2e}".format(x)},
        )
        .replace("[", "")
        .replace("]", "")
        .splitlines()
    )
    rv = [r"\begin{bmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)


def cast_int_to_float(inputs: BboxType) -> BboxType:
    """Converts int to float.

    Args:
        inputs (BboxType): Input bounding box.

    Returns:
        (BboxType): Bounding box with float type.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.float()
    return inputs.astype(np.float32)


def clone(inputs: BboxType) -> BboxType:
    """Clones bounding box.

    Note:
        copy.deepcopy() does not work for autograd tensors.

    Args:
        inputs (BboxType): Input bounding box.

    Returns:
        (BboxType): Cloned bounding box.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.clone()
    return inputs.copy()


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# pylint: disable=too-many-locals
def intersection_over_union(
    bbox_1: BboxType, bbox_2: BboxType, bbox_format: str = "yolo"
) -> torch.Tensor:
    """Calculates intersection over union between two bounding boxes.

    Note:
        The input bboxes can have any dimensions as long as the last dimension
        is 4, where the 4 elements are the coordinates of the bounding box.
        Therefore, in the case of YOLOv1, the input bboxes can be of shape like
        (16, 7, 7, 4) where 16 is the batch size, 7 is the grid size, and 4 is
        the coordinates of the bounding box.

        Since the grid size is 7, it means that there are 7 x 7 = 49 pairs of bounding boxes
        to compare.

        The above will return a tensor of shape (16, 7, 7, 1) where 16 is the
        batch size and for each batch, we have a 7 x 7 grid of IoU values for
        each pair of bounding boxes in the grid cell. Note that the last dimension
        1 can be "ignored" as it is an artifact of the broadcasting on the input
        shape of (16, 7, 7, 4).

    Args:
        bbox_1 (BboxType): Bounding Box 1 of shape (1, 4).
        bbox_2 (BboxType): Bounding Box 2 of shape (1, 4).
        bbox_format (str): The format of the bounding boxes. Default is "yolo" and
            must be one of the following: ["yolo", "voc", "albu"]

    Returns:
        iou (torch.Tensor): Intersection over union for each pair of bounding boxes.
            The shape is a scalar.

    References:
        Code modified from https://github.com/aladdinpersson/Machine-Learning-Collection
    """
    assert bbox_format in ["yolo", "voc", "albu"]

    if isinstance(bbox_1, np.ndarray):
        bbox_1 = torch.from_numpy(bbox_1)

    if isinstance(bbox_2, np.ndarray):
        bbox_2 = torch.from_numpy(bbox_2)

    if bbox_format == "yolo":
        bbox_1 = yolo2albu(bbox_1)
        bbox_2 = yolo2albu(bbox_2)

    # at this stage bbox must be in either albu or voc format.
    box1_x1 = bbox_1[..., 0:1]
    box1_y1 = bbox_1[..., 1:2]
    box1_x2 = bbox_1[..., 2:3]
    box1_y2 = bbox_1[..., 3:4]
    box2_x1 = bbox_2[..., 0:1]
    box2_y1 = bbox_2[..., 1:2]
    box2_x2 = bbox_2[..., 2:3]
    box2_y2 = bbox_2[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
    # squeeze all dimensions to scalar
    iou = iou.item()

    return iou


def non_max_suppression(
    bboxes: Union[torch.Tensor, list],
    iou_threshold: float,
    obj_threshold: float,
    bbox_format: str = "yolo",
):
    """Perform NMS on bounding boxes.

    Shape:
        bboxes: (N, 6) where N is the number of bounding boxes in the image
                       and 6 is [class_id, obj_conf, 4 coordinates depending on bbox_format]
                       and note the sequence must follow closely to the output of your
                       decode function.

        bboxes_after_nms: (N, 6) where N is the number of bounding boxes after NMS (remaining).


    Args:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        obj_threshold (float): threshold to remove predicted bboxes (independent of IoU)
        bbox_format (str): "midpoint" or "corners" used to specify bboxes (yolo vs pascal voc)

    Shape:
        bboxes: [S * S, 6] -> [S * S, [class_pred, prob_score, x1, y1, x2, y2]]
                If S = 7, then [49, 6]

    Returns:
        bboxes_after_nms (torch.Tensor): bboxes after performing NMS given a specific IoU threshold
    """

    # bboxes: [S * S, 6] -> [S * S, [class_id, conf, x1, y1, x2, y2]]
    bboxes = [bbox for bbox in bboxes if bbox[1] > obj_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_bbox = bboxes.pop(0)

        bboxes = [
            bbox
            for bbox in bboxes
            if bbox[0] != chosen_bbox[0]
            or intersection_over_union(
                torch.tensor(chosen_bbox[2:]),
                torch.tensor(bbox[2:]),
                bbox_format=bbox_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_bbox)

    if len(bboxes_after_nms) == 0:
        # hardcoded since we expect 6 values
        bboxes_after_nms = torch.empty((0, 6))
    else:
        bboxes_after_nms = torch.stack(bboxes_after_nms, dim=0)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    bbox_format="yolo",
    num_classes=20,
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    bbox_format=bbox_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def yolo2voc(inputs: BboxType, height: float, width: float) -> BboxType:
    """Converts from normalized [x, y, w, h] to [x1, y1, x2, y2] format.

    (x, y): the coordinates of the center of the bounding box;
    (w, h): the width and height of the bounding box.
    (x1, y1): the coordinates of the top left corner of the bounding box;
    (x2, y2): the coordinates of the bottom right corner of the bounding box.

    [x1, y1, x2, y2] is calculated as:
    x1 = x * width - w * width / 2
    y1 = y * height - h * height / 2
    x2 = x * width + w * width / 2
    y2 = y * height + h * height / 2

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) with the format
            `(center x, center y, width, height)` normalized by image width and
            height.
        height (float): Height of the image frame.
        width (float): Width of the image frame.

    Returns:
        outputs (BboxType): Bounding boxes of shape (..., 4) with the format `(top
            left x, top left y, bottom right x, bottom right y)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., [0, 2]] *= width
    outputs[..., [1, 3]] *= height

    outputs[..., 0] -= outputs[..., 2] / 2
    outputs[..., 1] -= outputs[..., 3] / 2
    outputs[..., 2] += outputs[..., 0]
    outputs[..., 3] += outputs[..., 1]

    return outputs


def yolo2albu(inputs: BboxType) -> BboxType:
    """Converts from normalized [x, y, w, h] to [x1, y1, x2, y2] normalized format.

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) with the format
            `(center x, center y, width, height)` normalized by image width and
            height.

    Returns:
        outputs (BboxType): Bounding boxes of shape (..., 4) with the format `(top
            left x, top left y, bottom right x, bottom right y)` normalized by
            image width and height.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 0] -= outputs[..., 2] / 2
    outputs[..., 1] -= outputs[..., 3] / 2
    outputs[..., 2] += outputs[..., 0]
    outputs[..., 3] += outputs[..., 1]

    return outputs
