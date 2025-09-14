import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1Darknet
from train import train_one_epoch, valid_one_epoch
from dataset import VOCDataset, get_transform
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    seed_all,
)
from loss import YoloLoss

DEBUG = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # "mps" # if macos m1
print(f"Using {DEVICE}")

# Hyperparameters etc.
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
NUM_WORKERS = 0
seed_all(seed=1992)


def main(debug: bool = True):
    if debug:
        BATCH_SIZE = 4
        EPOCHS = 10
    else:
        BATCH_SIZE = 4
        EPOCHS = 100

    S, B, C = 7, 2, 20
    model = Yolov1Darknet(
        in_channels=3, grid_size=7, num_bboxes_per_grid=2, num_classes=20
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S, B, C)

    ### Load Data ###
    csv_file = "./datasets/pascal_voc_128/pascal_voc_128.csv"
    images_dir = "./datasets/pascal_voc_128/images"
    labels_dir = "./datasets/pascal_voc_128/labels"

    ### Transforms ###
    train_transforms = get_transform(mode="train")
    valid_transforms = get_transform(mode="valid")

    voc_dataset_train = VOCDataset(
        csv_file,
        images_dir,
        labels_dir,
        train_transforms,
        S,
        B,
        C,
        mode="train",
    )
    voc_dataset_valid = VOCDataset(
        csv_file,
        images_dir,
        labels_dir,
        valid_transforms,
        S,
        B,
        C,
        mode="valid",
    )

    if debug:
        # remember to convert to list as __getitem__ takes in index as type int
        subset_indices = torch.arange(32)
        # purposely pick easy images for the 1st batch to illustrate for audience
        subset_indices[1] = 10
        subset_indices[2] = 12
        subset_indices[3] = 18
        subset_indices = subset_indices.tolist()

        voc_dataset_debug = torch.utils.data.Subset(
            voc_dataset_train, subset_indices
        )
        voc_dataloader_debug = DataLoader(
            dataset=voc_dataset_debug,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    voc_dataloader_train = DataLoader(
        dataset=voc_dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    voc_dataloader_valid = DataLoader(
        dataset=voc_dataset_valid,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        # mean_avg_prec = mean_average_precision(
        #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        # )
        # print(f"epoch: {epoch} Train mAP: {mean_avg_prec}")

        # if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        if debug:
            train_one_epoch(
                voc_dataloader_debug, model, optimizer, loss_fn, epoch, DEVICE
            )
            valid_one_epoch(
                voc_dataloader_debug, model, optimizer, loss_fn, epoch, DEVICE
            )
        else:
            train_one_epoch(
                voc_dataloader_train, model, optimizer, loss_fn, epoch, DEVICE
            )
            valid_one_epoch(
                voc_dataloader_valid, model, optimizer, loss_fn, epoch, DEVICE
            )


if __name__ == "__main__":
    main(debug=DEBUG)
