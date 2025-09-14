"""
Implementation of Yolo Loss Function from the original yolo paper
"""
import torch
from torch import nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model following https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py
    """

    def __init__(self, S: int = 7, B: int = 2, C: int = 20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def _get_bounding_box_coordinates_loss(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        better_bbox_index: int,
        vectorized_obj_indicator_ij: torch.Tensor,
    ) -> torch.Tensor:
        """FOR BOX COORDINATES corresponding to line 1-2 of the paper.

        Note:
            This loss calculates the loss for width and height of the bounding box.
            This loss calculates the loss for the center coordinates of the bounding box.

        Args:
            y_trues (torch.Tensor): The true bounding box coordinates.
            y_preds (torch.Tensor): The predicted bounding box coordinates.
            better_bbox_index (int): The index of the predicted bounding box with higher iou.
            vectorized_obj_indicator_ij (torch.Tensor): 1_{ij}^{obj} in paper refers to the jth bounding box predictor in cell i is responsible for that prediction.

        Returns:
            box_loss (torch.Tensor): The loss for the bounding box coordinates.
        """

        # Note that vectorized_obj_indicator_ij is vectorized and we one-hot encode it such that
        # grid cells with no object is 0.
        # We only take out one of the two y_preds, which is the one with highest Iou calculated previously.
        box_y_preds = vectorized_obj_indicator_ij * (
            (
                better_bbox_index * y_preds[..., 26:30]
                + (1 - better_bbox_index) * y_preds[..., 21:25]
            )
        )

        box_y_trues = vectorized_obj_indicator_ij * y_trues[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_y_preds[..., 2:4] = torch.sign(box_y_preds[..., 2:4]) * torch.sqrt(
            torch.abs(box_y_preds[..., 2:4] + 1e-6)
        )
        box_y_trues[..., 2:4] = torch.sqrt(box_y_trues[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_y_preds, end_dim=-2),
            torch.flatten(box_y_trues, end_dim=-2),
        )
        return box_loss

    def _get_objectness_loss(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        better_bbox_index: int,
        vectorized_obj_indicator_ij: torch.Tensor,
    ) -> torch.Tensor:
        """Corresponds to row 3 in the paper."""
        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            better_bbox_index * y_preds[..., 25:26]
            + (1 - better_bbox_index) * y_preds[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(vectorized_obj_indicator_ij * pred_box),
            torch.flatten(vectorized_obj_indicator_ij * y_trues[..., 20:21]),
        )
        return object_loss

    def _get_no_objectness_loss(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        vectorized_obj_indicator_ij: torch.Tensor,
    ) -> torch.Tensor:
        """Corresponds to row 4 in the paper."""

        no_object_loss = self.mse(
            torch.flatten(
                (1 - vectorized_obj_indicator_ij) * y_preds[..., 20:21],
                start_dim=1,
            ),
            torch.flatten(
                (1 - vectorized_obj_indicator_ij) * y_trues[..., 20:21],
                start_dim=1,
            ),
        )

        no_object_loss += self.mse(
            torch.flatten(
                (1 - vectorized_obj_indicator_ij) * y_preds[..., 25:26],
                start_dim=1,
            ),
            torch.flatten(
                (1 - vectorized_obj_indicator_ij) * y_trues[..., 20:21],
                start_dim=1,
            ),
        )

        return no_object_loss

    def _get_class_loss(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        vectorized_obj_indicator_ij: torch.Tensor,
    ) -> torch.Tensor:
        """Corresponds to row 5 in the paper."""

        class_loss = self.mse(
            torch.flatten(
                vectorized_obj_indicator_ij * y_preds[..., :20],
                end_dim=-2,
            ),
            torch.flatten(
                vectorized_obj_indicator_ij * y_trues[..., :20],
                end_dim=-2,
            ),
        )
        return class_loss

    def forward(
        self, y_preds: torch.Tensor, y_trues: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the loss for the yolo model.

        Expected Shapes:
            y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
            y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
        """

        y_preds = y_preds.reshape(-1, self.S, self.S, self.C + self.B * 5)
        assert y_preds.shape == y_trues.shape

        # note: in the 7x7x30 tensor, recall that there are grid cells with no bbox and hence are
        # initialized with 0s; as a result, the model will predict nonsense and possible negative values for these y_preds.

        # Calculate IoU for the two predicted bounding boxes with y_trues bbox:
        # iou_b1 = iou of bbox 1 and y_trues bbox
        # iou_b2 = iou of bbox 2 and y_trues bbox
        iou_b1 = intersection_over_union(
            y_preds[..., 21:25], y_trues[..., 21:25], bbox_format="yolo"
        )
        iou_b2 = intersection_over_union(
            y_preds[..., 26:30], y_trues[..., 21:25], bbox_format="yolo"
        )
        # print(iou_b1.shape, iou_b1[0])
        # iou_b3 = calculate_iou(y_preds[..., 26:30], y_trues[..., 21:25])
        # print(f"iou_b1: {iou_b1}")
        # print(f"iou_b2: {iou_b2}")
        # print(f"iou_b3: {iou_b3}")
        # concatenate the iou_b1 and iou_b2 tensors into a tensor array.
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # torch.max returns a namedtuple with the max value and its index.
        # _iou_max_score: out of the 2 predicted bboxes, _iou_max_score is the max score of the two;
        # better_bbox_index: the index of the better bbox (the one with the higher iou), either bbox 1 or bbox 2 (0 or 1 index).
        _iou_max_score, better_bbox_index = torch.max(ious, dim=0)

        # here aladdin vectorized 1_{ij}^{obj} and we name it vectorized_obj_indicator_ij
        vectorized_obj_indicator_ij = y_trues[..., 20].unsqueeze(3)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        box_loss = self._get_bounding_box_coordinates_loss(
            y_trues, y_preds, better_bbox_index, vectorized_obj_indicator_ij
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        object_loss = self._get_objectness_loss(
            y_trues, y_preds, better_bbox_index, vectorized_obj_indicator_ij
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self._get_no_objectness_loss(
            y_trues, y_preds, vectorized_obj_indicator_ij
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self._get_class_loss(
            y_trues, y_preds, vectorized_obj_indicator_ij
        )
        # print(self.lambda_coord * box_loss)
        # print(object_loss)
        # print(self.lambda_noobj * no_object_loss)
        # print(class_loss)

        total_loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return total_loss


# class YOLOv1Loss(nn.Module):
#     def __init__(
#         self,
#         S: int,
#         B: int,
#         C: int,
#         lambda_coord: float = 5,
#         lambda_noobj: float = 0.5,
#     ) -> None:
#         super().__init__()
#         self.S = S
#         self.B = B
#         self.C = C
#         self.lambda_coord = lambda_coord
#         self.lambda_noobj = lambda_noobj

#         # bbox loss
#         self.bbox_xy_offset_loss = 0
#         self.bbox_wh_loss = 0

#         # objectness loss
#         self.object_loss = 0
#         self.no_object_loss = 0

#         # class loss
#         self.class_loss = 0

#     def forward(self, y_trues: torch.Tensor, y_preds: torch.Tensor):
#         # y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
#         # y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
#         # however y_preds was flattened to (batch_size, S*S*(C + B * 5)) = (batch_size, 1470)
#         # so we need to reshape it back to (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)

#         y_preds = y_preds.reshape(-1, self.S, self.S, self.C + self.B * 5)
#         assert y_preds.shape == y_trues.shape

#         batch_size = y_preds.shape[0]

#         # if dataloader has a batch size of 2, then our total loss is the average of the two losses.
#         # i.e. total_loss = (loss1 + loss2) / 2 where loss1 is the loss for the first image in the
#         # batch and loss2 is the loss for the second image in the batch.

#         # to calculate total loss for each image we use the following formula:

#         for batch_index in range(batch_size):

#             for row in range(self.S):
#                 for col in range(self.S):
#                     # this double loop is like matrix: if a matrix
#                     # M is of shape (S, S) = (7, 7) then this double loop is
#                     # M_ij where i is the row and j is the column.
#                     # so first loop is M_{11}, second loop is M_{12}...

#                     # check 20 suffice cause by construction both index 20 and 25 will be filled with
#                     # the same objectness score (0 or 1)
#                     has_object = y_trues[:, row, col, 20] == 1
#                     if has_object:
#                         # 1_obj means if has object then calculate else 0

#                         y_preds_bbox1 = 1
#                         # x_grid_offset_true = y_trues[:, row, col, ]
#                         # x_grid_offset_pred
#                         # grid_bbox_xy_offset_loss =

#                         #                         pred_bbox1 = torch.Tensor(
#                         #     [preds[i, y, x, 0], preds[i, y, x, 1], preds[i, y, x, 2], preds[i, y, x, 3]])
#                         # pred_bbox2 = torch.Tensor(
#                         #     [preds[i, y, x, 5], preds[i, y, x, 6], preds[i, y, x, 7], preds[i, y, x, 8]])
#                         # label_bbox = torch.Tensor(
#                         #     [labels[i, y, x, 0], labels[i, y, x, 1], labels[i, y, x, 2], labels[i, y, x, 3]])

#         return

#         # y_trues shape: (1, 7, 7, 30)
#         # y_preds shape: (1, 7, 7, 30)
#         # where the 30 elements are
#         # [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20,
#         #  x_grid, y_grid, w, h, objectness,
#         #  x_grid, y_grid, w, h, objectness]
