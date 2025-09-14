"""Points to note:
1. You need to use collate_fn if you are returning bboxes straight: https://discuss.pytorch.org/t/
   dataloader-collate-fn-throws-runtimeerror-stack-expects-each-tensor-to-be-equal-size-in-response-
   to-variable-number-of-bounding-boxes/117952/3

2. Train and validation column in csv is hardcoded, in actual training need to be random split.
3. Note that this implementation assumes S=7, B=2, C=20 and if you change then the code will break since
tensor slicing are performed based on this assumption.
4. Assumes train df has a column called train_flag.
5. If diff num of classes, need to recode the encode and decode...
"""
import os
from typing import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as FT

from PIL import Image
from torch.utils.data import DataLoader


# FIXME: see the mzbai's repo to collate fn properly so can use albu!
def get_transform(mode: str, image_size: int = 448) -> T.Compose:
    transforms = []
    # transforms.append(T.PILToTensor())
    # this is must need if not will have error or use TOTensor.
    # transforms.append(T.ConvertImageDtype(torch.float))
    transforms.append(T.Resize((image_size, image_size)))
    transforms.append(T.ToTensor())

    # transforms.append(
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # )

    if mode == "train":
        # do nothing for now as we want to ensure there is not flipping.
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass

    return T.Compose(transforms)


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file: str,
        images_dir: str,
        labels_dir: str,
        transforms: T.Compose,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        mode: str = "train",
    ) -> None:
        self.csv_file = csv_file
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # train/valid/test dataframe
        self.df = self.get_df()

    def get_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_file)
        return df[df["train_flag"] == self.mode].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def strip_label_path(label_path: str) -> List[list]:
        bboxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                bboxes.append([class_label, x, y, width, height])

        return bboxes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(
            self.images_dir, self.df.loc[index, "image_id"]
        )
        image = Image.open(image_path).convert("RGB")

        label_path = os.path.join(
            self.labels_dir, self.df.loc[index, "label_id"]
        )
        bboxes = self.strip_label_path(label_path)

        if self.transforms:
            image = self.transforms(image)
            bboxes = torch.tensor(bboxes, dtype=torch.float)

        bboxes = encode(bboxes, self.S, self.B, self.C)

        return image, bboxes


def encode(bboxes: torch.Tensor, S: int, B: int, C: int) -> torch.Tensor:
    label_matrix = torch.zeros((S, S, 5 * B + C))

    for bbox in bboxes:
        class_id = int(bbox[0])
        x_center, y_center, width, height = bbox[1:]

        x_grid = int(torch.floor(S * x_center))
        y_grid = int(torch.floor(S * y_center))

        x_grid_offset, y_grid_offset = (
            S * x_center - x_grid,
            S * y_center - y_grid,
        )

        if (
            label_matrix[y_grid, x_grid, 20] == 0
            and label_matrix[y_grid, x_grid, 25] == 0
        ):

            label_matrix[y_grid, x_grid, 20] = 1
            label_matrix[y_grid, x_grid, 25] = 1

            encoded_bbox_coordinates = torch.tensor(
                [x_grid_offset, y_grid_offset, width, height]
            )

            label_matrix[y_grid, x_grid, 21:25] = encoded_bbox_coordinates
            label_matrix[y_grid, x_grid, 26:30] = encoded_bbox_coordinates

            label_matrix[y_grid, x_grid, class_id] = 1

    return label_matrix


def decode(outputs: torch.Tensor, S: int = 7) -> torch.Tensor:
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """
    batch_size = outputs.shape[0]

    # outputs: either [bs, 7, 7, 30] or [bs, 1470] where 1470 = 7 * 7 * 30 flattened
    outputs = outputs.detach().cpu()
    outputs = outputs.reshape(batch_size, 7, 7, 30)

    # bbox_1: [bs, 7, 7, 4] where 4 is [x_grid_offset, y_grid_offset, width, height]
    bbox_1 = outputs[..., 21:25]
    bbox_2 = outputs[..., 26:30]

    # obj_scores: [2, bs, 7, 7] the bbox objectness confidence scores
    # logic: assume S=7, B=2, C=20, then a total of 98 bboxes are predicted
    # now he reshapes to [2, bs, 7, 7] where the first element is the scores for
    # the first box and the second element is the scores for the second box.
    # key: flatten this along the bs dim results in [2 * 49] = 98
    obj_scores = torch.cat(
        (outputs[..., 20].unsqueeze(0), outputs[..., 25].unsqueeze(0)),
        dim=0,
    )

    # best_bbox_index: [bs, 7, 7, 1]
    # logic: best_bbox_index is the index of the best box, either 0 or 1 in this case
    # since we have only 2 bboxes with index 0 and 1.
    # key: flatten this along the bs dim results in [49] = 49 which is correct since
    # we choose the best box for each cell, out of 49 cells.
    best_bbox_index = obj_scores.argmax(0).unsqueeze(-1)

    # best_bbox: [bs, 7, 7, 4]
    # after all the sophistication, best_bbox is either bbox_1 or bbox_2
    # depending on the best_bbox_index.
    # the formula below ensures that if best_bbox_index is 0, then best_bbox is bbox_1
    # as the bbox_2 -> 0 since best_bbox_index is 0.
    best_bbox = bbox_1 * (1 - best_bbox_index) + best_bbox_index * bbox_2

    # cell_indices: [bs, 7, 7, 1]
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    # x_grid_offset: [bs, 7, 7, 1]
    # logic: x_grid_offset is the x offset of the bbox relative to the cell
    # same for y_grid_offset
    # key: the logic is clearer now why he wanted to unsqueeze cell_indices to have
    # an additional dimension, this is for the broadcasting to work (add).
    x_grid_offset = best_bbox[..., :1]
    y_grid_offset = best_bbox[..., 1:2]

    # x_center: [bs, 7, 7, 1] | y_center: [bs, 7, 7, 1]
    # logic: we recovered the x_center and y_center from the original yolo format.
    x_center = 1 / S * (x_grid_offset + cell_indices)
    y_center = 1 / S * (y_grid_offset + cell_indices.permute(0, 2, 1, 3))

    # wh: [bs, 7, 7, 2]
    # logic: we recovered the width and height from the original yolo format.
    # and this remains unchanged anyways.
    wh = best_bbox[..., 2:4]

    # converted_bboxes: [bs, 7, 7, 4]
    converted_bboxes = torch.cat((x_center, y_center, wh), dim=-1)

    # FIXME: should here not be the class_id of the best bbox?
    # class_id: [bs, 7, 7, 1]
    class_id = outputs[..., :20].argmax(-1).unsqueeze(-1)

    # FIXME: should here not be the best_confidence of the best bbox?
    # best_confidence: [bs, 7, 7, 1]
    best_confidence = torch.max(outputs[..., 20], outputs[..., 25]).unsqueeze(
        -1
    )

    # detected_bboxes: [bs, 7, 7, 6]
    # logic: the 6 dimensions are [class_id, best_confidence, x_center, y_center, width, height]

    decoded_bbox = torch.cat(
        (class_id, best_confidence, converted_bboxes), dim=-1
    )

    # logic: the final form here assumes that we have only 1 bbox per cell, hence [bs, 49, 6]
    flattened_decoded_bbox = flatten_decoded_bbox(decoded_bbox, S)
    return flattened_decoded_bbox


def flatten_decoded_bbox(decoded_bbox: torch.Tensor, S: int) -> torch.Tensor:
    """Flattens the decoded bbox to [bs, S * S, 6].

    Args:
        decoded_bbox (torch.Tensor): The decoded bbox of shape [bs, S, S, 6].
        S (int): The grid size.

    Returns:
        flattened_decoded_bbox (torch.Tensor): The flattened decoded bbox of shape [bs, S * S, 6].
    """

    # flattened_decoded_bbox: [bs, 7 * 7, 6] = [bs, 49, 6]
    flattened_decoded_bbox = decoded_bbox.reshape(
        decoded_bbox.shape[0], S * S, -1
    )
    # turn class_id to int
    flattened_decoded_bbox[..., 0] = flattened_decoded_bbox[..., 0].long()

    return flattened_decoded_bbox


def load_debug_dataloader():
    csv_file = "./datasets/pascal_voc_128/pascal_voc_128.csv"
    images_dir = "./datasets/pascal_voc_128/images"
    labels_dir = "./datasets/pascal_voc_128/labels"

    # remember to convert to list as __getitem__ takes in index as type int
    subset_indices = torch.arange(32)
    # purposely pick easy images for the 1st batch to illustrate for audience
    subset_indices[1], subset_indices[2], subset_indices[3] = 10, 12, 18
    subset_indices = subset_indices.tolist()

    S, B, C = 7, 2, 20
    valid_transforms = get_transform(mode="valid")

    voc_dataset_train = VOCDataset(
        csv_file,
        images_dir,
        labels_dir,
        valid_transforms,
        S,
        B,
        C,
        mode="train",
    )

    voc_dataset_debug = torch.utils.data.Subset(
        voc_dataset_train, subset_indices
    )
    voc_dataloader_debug = DataLoader(
        dataset=voc_dataset_debug,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return voc_dataloader_debug


def plot_images_from_dataloader(
    inputs: Union[torch.Tensor, List[torch.Tensor]]
):
    image_grid = torchvision.utils.make_grid(inputs)
    _fig = plt.figure(figsize=(30, 30))
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    csv_file = "./datasets/pascal_voc_128/pascal_voc_128.csv"
    images_dir = "./datasets/pascal_voc_128/images"
    labels_dir = "./datasets/pascal_voc_128/labels"

    S, B, C = 7, 2, 20
    mode = "train"
    train_transforms = get_transform(mode=mode)

    voc_dataset_train = VOCDataset(
        csv_file, images_dir, labels_dir, train_transforms, S, B, C, mode
    )

    print(f"Length of the dataset: {len(voc_dataset_train)}")

    for image, bboxes in voc_dataset_train:
        # print(bboxes)
        print(f"type of image: {type(image)}, type of bboxes: {type(bboxes)}")
        print(
            f"shape of image: {image.shape}, shape of bboxes: {bboxes.shape}"
        )
        print(f"bboxes: {bboxes}")

        break

    BATCH_SIZE = 4
    NUM_WORKERS = 0
    PIN_MEMORY = True
    SHUFFLE = False
    DROP_LAST = True

    train_loader = torch.utils.data.DataLoader(
        dataset=voc_dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
    )

    for batch_index, (images, bboxes) in enumerate(train_loader):

        images = images.detach().cpu()
        bboxes = bboxes.detach().cpu()

        print(f"images shape: {images.shape}, bboxes shape: {bboxes.shape}")

        decoded_bboxes = decode(bboxes)
        print(f"decoded bboxes: {decoded_bboxes.shape}")
        print(f"decoded bboxes: {decoded_bboxes}")

        voc_bboxes = yolo2voc(decoded_bboxes[..., 2:], height=448, width=448)
        print(f"voc bboxes: {voc_bboxes.shape}")
        print(f"voc bboxes: {voc_bboxes}")

        image_grid = []
        for image, voc_bbox in zip(images, voc_bboxes):
            image = torch.from_numpy(
                np.asarray(FT.to_pil_image(image))
            ).permute(2, 0, 1)
            overlayed_image = torchvision.utils.draw_bounding_boxes(
                image,
                voc_bbox,
                colors=["red"] * 49,
                # labels=["dog"] * 49,
                width=6,
            )
            image_grid.append(overlayed_image)

        grid = torchvision.utils.make_grid(image_grid)
        # print(f"shape of overlayed_images: {overlayed_images.shape}")
        fig = plt.figure()
        plt.imshow(grid.numpy().transpose(1, 2, 0))
        # plt.imshow(grid.numpy().transpose((1, 2, 0)))
        # plt.savefig(os.path.join(output_path, "batch0.png"))
        plt.show()
        # plt.close(fig)
        break
