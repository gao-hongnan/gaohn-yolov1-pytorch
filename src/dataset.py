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
import torch
import os
import pandas as pd
from PIL import Image
from typing import Tuple
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
from utils import yolo2voc
from typing import List
import torchvision.transforms.functional as FT
import numpy as np
from torch.utils.data import DataLoader
from utils import bmatrix

# FIXME: see the mzbai's repo to collate fn properly so can use albu!
def get_transform(mode: str, image_size: int = 448) -> T.Compose:
    """Create a torchvision transform for the given mode.

    Note:
        You can append more transforms to the list if you want.
        For simplicity of this exercise, we will hardcode the transforms.

    Args:
        mode (str): The mode of the dataset, one of [train, valid, test].
        image_size (int, optional): The image size to resize to. Defaults to 448.

    Returns:
        T.Compose: The torchvision transform pipeline.
    """
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
        """Dataset class for Pascal VOC format.

        Args:
            csv_file (str): The path of the csv.
            images_dir (str): The path of the images directory.
            labels_dir (str): The path of the labels directory.
            transforms (_type_): The transform function
            S (int): Grid Size. Defaults to 7.
            B (int): Number of Bounding Boxes per Grid. Defaults to 2.
            C (int): Number of Classes. Defaults to 20.
            mode (str): The mode of the dataset. Defaults to "train". Must be one of
                        ["train", "valid", "test"]
        """

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
        """This method returns the train/valid/test dataframe according to the mode."""

        df = pd.read_csv(self.csv_file)
        return df[df["train_flag"] == self.mode].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def strip_label_path(label_path: str) -> List[list]:
        """Strips the label path and returns the bounding boxes
        in the format of [class_id, x, y, width, height] in yolo format.

        Args:
            label_path (str): The path of the labels.

        Returns:
            bbox (List[list]): The bounding boxes in the format of
                               [class_id, x, y, width, height] in yolo format.
        """
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
        """Returns the image and the bboxes tensor for the given index.

        Returns:
            image (torch.Tensor): Image tensor of shape (3, image_size, image_size).
            bboxes (torch.Tensor): Bounding boxes tensor of shape (S, S, 5 * B + C).
        """
        image_path = os.path.join(self.images_dir, self.df.loc[index, "image_id"])

        image = Image.open(image_path).convert("RGB")

        label_path = os.path.join(self.labels_dir, self.df.loc[index, "label_id"])

        bboxes = self.strip_label_path(label_path)

        if self.transforms:
            # image: (3, image_size, image_size)
            image = self.transforms(image)
            # bboxes: [num_bbox, 5] where 5 is [class_id, x, y, width, height] yolo format
            bboxes = torch.tensor(bboxes, dtype=torch.float)

        bboxes = encode(bboxes, self.S, self.B, self.C)
        return image, bboxes


def encode(bboxes: torch.Tensor, S: int, B: int, C: int) -> torch.Tensor:
    """Convert bounding boxes from xywh to label matrix for ingestion by Yolo v1.

    Following convention:
    - https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/amp/ and
    - https://towardsdatascience.com/yolov1-you-only-look-once-object-detection-e1f3ffec8a89

    Label matrix is 7x7x30 where the depth 30 is:
    [x_grid, y_grid, w, h, objectness, x_grid, y_grid, w, h, objectness, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]
    where p1-p20 are the 20 classes.

    But we follow aladdinpersson https://github.com/aladdinpersson/Machine-Learning-Collection/blob/
                                master/ML/Pytorch/object_detection/YOLO/dataset.py where we follow the reverse convention:
    [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, objectness, x_grid, y_grid, w, h, objectness, x_grid, y_grid, w, h]

    Args:
        bboxes (torch.Tensor): bboxes in YOLO format (class_label, x_center, y_center, width, height)
                               where coordinates are normalized to [0, 1].

    Returns:
        label_matrix (torch.Tensor): label matrix in YOLO format.
    """

    # initialize label_matrix
    # (S, S, 5 * B + C) -> (S, S, 30) if S=7, B=2, C=20
    label_matrix = torch.zeros((S, S, 5 * B + C))

    # bboxes: [num_bbox, 5] where 5 is [class_id, x, y, width, height] yolo format
    for bbox in bboxes:

        # unpack yolo bbox
        class_id = int(bbox[0])
        x_center, y_center, width, height = bbox[1:]

        x_grid = int(torch.floor(S * x_center))  # 当前bbox中心落在第gridx个网格,列
        y_grid = int(torch.floor(S * y_center))  # 当前bbox中心落在第gridy个网格,行

        # TODO: annotate in my documentation on exactly what this is.
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        x_grid_offset, y_grid_offset = (
            S * x_center - x_grid,
            S * y_center - y_grid,
        )

        # print(f"class_id: {class_id}")
        # print(f"x_center: {x_center} y_center: {y_center}")
        # print(f"width: {width} height: {height}")
        # print(f"x_grid: {x_grid} y_grid: {y_grid}")
        # print(f"x_grid_offset: {x_grid_offset} y_grid_offset: {y_grid_offset}")

        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        # here we fill both bbox's objectness to be 1 if it is originally 0
        if (
            label_matrix[y_grid, x_grid, 4] == 0
            and label_matrix[y_grid, x_grid, 9] == 0
        ):
            # [conf, x_grid_offset, y_grid_offset, width, height]
            encoded_bbox_coordinates = torch.tensor(
                [x_grid_offset, y_grid_offset, width, height, 1]
            )

            label_matrix[y_grid, x_grid, 0:5] = encoded_bbox_coordinates
            label_matrix[y_grid, x_grid, 5:10] = encoded_bbox_coordinates

            # set class probability to 1 at the class_id index shifted by 5 * B
            label_matrix[y_grid, x_grid, 5 * B + class_id] = 1

    # (7, 7, 30) -> (49, 30)
    label_matrix = label_matrix.reshape(-1, S * S, 5 * B + C)
    return label_matrix


def decode(outputs: torch.Tensor, S: int = 7, B: int = 2, C: int = 20) -> torch.Tensor:
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
    outputs = outputs.reshape(batch_size, S, S, 5 * B + C)

    # bbox_1: [bs, 7, 7, 4] where 4 is [x_grid_offset, y_grid_offset, width, height]
    bbox_1 = outputs[..., 0:4]
    bbox_2 = outputs[..., 5:9]

    # obj_scores: [2, bs, 7, 7] the bbox objectness confidence scores
    # logic: assume S=7, B=2, C=20, then a total of 98 bboxes are predicted
    # now he reshapes to [2, bs, 7, 7] where the first element is the scores for
    # the first box and the second element is the scores for the second box.
    # key: flatten this along the bs dim results in [2 * 49] = 98
    obj_scores = torch.cat(
        (outputs[..., 4].unsqueeze(0), outputs[..., 9].unsqueeze(0)),
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
    x_grid_offset = best_bbox[
        ..., 0:1
    ]  # best_bbox[..., :1] changed cause of encode changed
    y_grid_offset = best_bbox[..., 1:2]  # best_bbox[..., 1:2]

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
    class_id = outputs[..., 10:].argmax(-1).unsqueeze(-1)

    # FIXME: should here not be the best_confidence of the best bbox?
    # best_confidence: [bs, 7, 7, 1]
    best_confidence = torch.max(outputs[..., 4], outputs[..., 9]).unsqueeze(-1)

    # detected_bboxes: [bs, 7, 7, 6]
    # logic: the 6 dimensions are [class_id, best_confidence, x_center, y_center, width, height]

    decoded_bbox = torch.cat((class_id, best_confidence, converted_bboxes), dim=-1)

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
    flattened_decoded_bbox = decoded_bbox.reshape(decoded_bbox.shape[0], S * S, -1)
    # turn class_id to int
    flattened_decoded_bbox[..., 0] = flattened_decoded_bbox[..., 0].long()

    return flattened_decoded_bbox


# def decode(label_matrix: torch.Tensor, S: int, B: int, C: int) -> torch.Tensor:
#     """Convert label matrix to bounding boxes in xywh format.

#     Args:
#         label_matrix (torch.Tensor): label matrix in YOLO format.

#     Returns:
#         bboxes (torch.Tensor): bboxes in YOLO format (class_label, x_center, y_center, width, height)
#                                where coordinates are normalized to [0, 1].
#     """
#     bboxes = []

#     # (S, S, 5 * B + C) -> (S, S, 30) if S=7, B=2, C=20
#     for i, j in itertools.product(range(S), range(S)):

#         # check if objectness is 1
#         if label_matrix[i, j, 20] == 1 or label_matrix[i, j, 25] == 1:

#             # get class probabilities
#             class_probabilities = label_matrix[i, j, :20]

#             # get class with highest probability
#             class_id = torch.argmax(class_probabilities)

#             # get bbox coordinates
#             bbox_coordinates = (
#                 label_matrix[i, j, 21:25]
#                 if label_matrix[i, j, 20] == 1
#                 else label_matrix[i, j, 26:30]
#             )

#             # unpack bbox coordinates
#             x_grid_offset, y_grid_offset, width, height = bbox_coordinates

#             # convert bbox coordinates to xywh
#             x_center = (j + x_grid_offset) / S
#             y_center = (i + y_grid_offset) / S

#             bboxes.append(
#                 torch.tensor([class_id, x_center, y_center, width, height])
#             )

#     return torch.stack(bboxes)


def get_debug_dataset():
    csv_file = "./datasets/pascal_voc_128/pascal_voc_128.csv"
    images_dir = "./datasets/pascal_voc_128/images"
    labels_dir = "./datasets/pascal_voc_128/labels"

    S, B, C = 7, 2, 20

    ### Transforms ###
    no_transforms = get_transform(mode="valid")

    voc_dataset_train_no_transforms = VOCDataset(
        csv_file,
        images_dir,
        labels_dir,
        no_transforms,
        S,
        B,
        C,
        mode="train",
    )

    # remember to convert to list as __getitem__ takes in index as type int
    subset_indices = torch.arange(32)
    # purposely pick easy images for the 1st batch to illustrate for audience
    subset_indices[1] = 10
    subset_indices[2] = 12
    subset_indices[3] = 18
    subset_indices = subset_indices.tolist()

    voc_dataset_debug = torch.utils.data.Subset(
        voc_dataset_train_no_transforms, subset_indices
    )
    return voc_dataset_debug


if __name__ == "__main__":
    image_1_yolo_label = VOCDataset.strip_label_path(
        "./datasets/pascal_voc_128/labels/000001.txt"
    )
    print(bmatrix(image_1_yolo_label))
    voc_dataset_debug = get_debug_dataset()

    print(f"Length of the dataset: {len(voc_dataset_debug)}")

    for image, bboxes in voc_dataset_debug:
        # print(bboxes)
        print(f"type of image: {type(image)}, type of bboxes: {type(bboxes)}")
        print(f"shape of image: {image.shape}, shape of bboxes: {bboxes.shape}")
        print(f"bboxes: {bboxes}")

        break

    voc_dataloader_debug = DataLoader(
        dataset=voc_dataset_debug,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    for batch_index, (images, bboxes) in enumerate(voc_dataloader_debug):

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
            image = torch.from_numpy(np.asarray(FT.to_pil_image(image))).permute(
                2, 0, 1
            )
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
