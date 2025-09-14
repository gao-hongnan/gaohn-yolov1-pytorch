"""
Main file for training Yolo model on Pascal VOC dataset
"""
import torch
import torchvision.transforms.functional as FT
from tqdm import tqdm

import os
import sys

sys.path.insert(1, os.getcwd())
from utils import non_max_suppression, yolo2voc
from typing import *
import time
from dataset import decode
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from config import config
from torch.utils.data import DataLoader
from model import Yolov1Darknet
from loss import YoloLoss

ClassMap = config.ClassMap()

plt.rcParams["savefig.bbox"] = "tight"


def train_one_epoch(
    train_loader: DataLoader,
    model: Yolov1Darknet,
    optimizer: torch.optim,
    criterion: YoloLoss,
    epoch: int,
    device: str,
) -> List[float]:
    """Train the model for one epoch.

    Args:
        train_loader (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        epoch (_type_): _description_

    Shapes:
        inputs: (batch_size, 3, 448, 448)
        y_trues: (batch_size, 7, 7, 30)
        y_preds: (batch_size, 7 * 7 * 30) = (batch_size, 1470)

    Note:
        It is worth noting that y_preds is reshaped from (batch_size, 7, 7, 30) to (batch_size, 7 * 7 * 30).

    Returns:
        List[float]: _description_
    """
    # torch.Size([16, 3, 448, 448]) torch.Size([16, 7, 7, 30])
    model.train()
    start = time.time()
    train_bar = tqdm(train_loader, leave=True)
    train_loss_epoch_history = []
    train_loss = 0

    for batch_idx, (inputs, y_trues) in enumerate(train_bar):
        # inputs: (batch_size, 3, 448, 448)
        # y_trues: (batch_size, 7, 7, 30)
        inputs, y_trues = inputs.to(device), y_trues.to(device)

        # y_preds: (batch_size, 7 * 7 * 30) -> (batch_size, 1470)
        y_preds = model(inputs)

        # y_trues_decoded: (batch_size, 7, 7, 6) -> (batch_size, 7 * 7, 6)
        # [class_id, obj_conf, x, y, w, h]
        y_trues_decoded = decode(y_trues.detach().cpu())
        y_preds_decoded = decode(y_preds.detach().cpu())

        loss = criterion(y_preds, y_trues)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # update progress bar
        train_bar.set_postfix(loss=loss.item())

    # if want add grid of images see:  https://github.com/ivanwhaf/yolov1-pytorch/blob/b7df7740bfa9326f3d84b7c10b4ec4ee03d607c0/train.py
    # TODO: add a function to plot image grids
    average_train_loss_per_epoch = train_loss / len(train_loader)
    print(f"Train Epoch: {epoch}")
    print(f"Mean loss: {average_train_loss_per_epoch}")
    print(f"Time Spent: {time.time() - start}s")
    train_loss_epoch_history.append(average_train_loss_per_epoch)

    return train_loss_epoch_history


@torch.inference_mode(mode=True)
def valid_one_epoch(
    valid_loader, model, optimizer, criterion, epoch, device, nms: bool = True
) -> List[float]:
    # same as train_one_epoch
    model.eval()
    start = time.time()

    valid_bar = tqdm(valid_loader, leave=True)
    valid_loss_epoch_history = []
    valid_loss = 0

    for batch_idx, (inputs, y_trues) in enumerate(valid_bar):
        # inputs: (batch_size, 3, 448, 448)
        # y_trues: (batch_size, 7, 7, 30)
        inputs, y_trues = inputs.to(device), y_trues.to(device)

        optimizer.zero_grad()

        # y_preds: (batch_size, 7 * 7 * 30) -> (batch_size, 1470)
        y_preds = model(inputs)

        loss = criterion(y_preds, y_trues)
        valid_loss += loss.item()

        # update progress bar
        valid_bar.set_postfix(loss=loss.item())

        ### DECODE ###
        inputs = inputs.detach().cpu()
        # y_trues_decoded: (batch_size, 7, 7, 6) -> (batch_size, 7 * 7, 6)
        # [class_id, obj_conf, x, y, w, h] recall this is in yolo format
        # so need to convert to voc for plotting (easier)
        y_trues_decoded = decode(y_trues.detach().cpu())
        y_preds_decoded = decode(y_preds.detach().cpu())

        if batch_idx == 0:
            image_grid = []
            # TODO: HONGNAN: remember find a way to turn TOTENSOR back to proper uint8 image? how?
            for (input, y_true_decoded, y_pred_decoded) in zip(
                inputs, y_trues_decoded, y_preds_decoded
            ):
                # FIXME: find way to turn Tensor back to uint8 image without using this.
                # input: (3, 448, 448)
                input = torch.from_numpy(
                    np.asarray(FT.to_pil_image(input))
                ).permute(2, 0, 1)

                # nms_bbox_pred: (N, 6) where N is number of bboxes after nms
                #                       and 6 is [class, obj_conf, x, y, w, h]
                y_pred_decoded_nms = non_max_suppression(
                    y_pred_decoded,
                    iou_threshold=0.5,
                    obj_threshold=0.4,
                    bbox_format="yolo",
                )
                class_ids = y_pred_decoded_nms[:, 0]
                num_bboxes_after_nms = y_pred_decoded_nms.shape[0]

                if num_bboxes_after_nms == 0:
                    # if no bboxes after nms, then just plot the image
                    # but in order for consistency, we just pass empty colors
                    # and class names to torchvision.utils.draw_bounding_boxes
                    # so it can just plot the original image instead of using continue.
                    colors = []
                    class_names = []
                else:
                    class_names = [
                        ClassMap.classes_map[int(class_idx.item())]
                        for class_idx in class_ids
                    ]
                    colors = ["red"] * num_bboxes_after_nms

                font_path = "./07558_CenturyGothic.ttf"

                y_true_decoded_voc = yolo2voc(
                    y_true_decoded[..., 2:],
                    height=inputs.shape[2],
                    width=inputs.shape[3],
                )
                overlayed_image_true = torchvision.utils.draw_bounding_boxes(
                    input,
                    y_true_decoded_voc,
                    # colors=colors,
                    width=6,
                    # labels=class_names,
                )

                y_pred_decoded_nms_voc = yolo2voc(
                    y_pred_decoded_nms[..., 2:],
                    height=inputs.shape[2],
                    width=inputs.shape[3],
                )
                overlayed_image_pred = torchvision.utils.draw_bounding_boxes(
                    input,
                    y_pred_decoded_nms_voc,
                    colors=colors,
                    width=6,
                    labels=class_names,
                    font_size=20,
                    font=font_path,
                )
                image_grid.append(overlayed_image_true)
                image_grid.append(overlayed_image_pred)
            grid = torchvision.utils.make_grid(image_grid)

            # print(f"shape of overlayed_images: {overlayed_images.shape}")
            fig = plt.figure(figsize=(30, 30))
            plt.imshow(grid.numpy().transpose(1, 2, 0))

            plt.savefig(f"epoch_{epoch}_batch0.png", dpi=300)
            print("saved")
            # plt.show()

            # END DECODE

    average_valid_loss_per_epoch = valid_loss / len(valid_loader)
    print(f"Valid Epoch: {epoch}")
    print(f"Mean loss: {average_valid_loss_per_epoch}")
    print(f"Time Spent: {time.time() - start}")

    valid_loss_epoch_history.append(average_valid_loss_per_epoch)
    return average_valid_loss_per_epoch


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
