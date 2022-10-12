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
                better_bbox_index * y_preds[..., 5:9]
                + (1 - better_bbox_index) * y_preds[..., 0:4]
            )
        )

        box_y_trues = vectorized_obj_indicator_ij * y_trues[..., 0:4]

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
            better_bbox_index * y_preds[..., 9:10]
            + (1 - better_bbox_index) * y_preds[..., 4:5]
        )

        object_loss = self.mse(
            torch.flatten(vectorized_obj_indicator_ij * pred_box),
            torch.flatten(vectorized_obj_indicator_ij * y_trues[..., 4:5]),
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
                (1 - vectorized_obj_indicator_ij) * y_preds[..., 4:5],
                start_dim=1,
            ),
            torch.flatten(
                (1 - vectorized_obj_indicator_ij) * y_trues[..., 4:5],
                start_dim=1,
            ),
        )

        no_object_loss += self.mse(
            torch.flatten(
                (1 - vectorized_obj_indicator_ij) * y_preds[..., 9:10],
                start_dim=1,
            ),
            torch.flatten(
                (1 - vectorized_obj_indicator_ij) * y_trues[..., 4:5],
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
                vectorized_obj_indicator_ij * y_preds[..., 10:],
                end_dim=-2,
            ),
            torch.flatten(
                vectorized_obj_indicator_ij * y_trues[..., 10:],
                end_dim=-2,
            ),
        )
        return class_loss

    def forward(self, y_preds: torch.Tensor, y_trues: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the yolo model.

        Expected Shapes:
            y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
            y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
        """

        y_preds = y_preds.reshape(-1, self.S, self.S, self.C + self.B * 5)
        assert y_preds.shape == y_trues.shape
        # print(y_trues[0, 1, 2, 0:5][..., 1])
        # note: in the 7x7x30 tensor, recall that there are grid cells with no bbox and hence are
        # initialized with 0s; as a result, the model will predict nonsense and possible negative values for these y_preds.

        # Calculate IoU for the two predicted bounding boxes with y_trues bbox:
        # iou_b1 = iou of bbox 1 and y_trues bbox
        # iou_b2 = iou of bbox 2 and y_trues bbox
        iou_b1 = intersection_over_union(
            y_preds[..., 0:4], y_trues[..., 0:4], bbox_format="yolo"
        )
        iou_b2 = intersection_over_union(
            y_preds[..., 5:9], y_trues[..., 0:4], bbox_format="yolo"
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
        vectorized_obj_indicator_ij = y_trues[..., 4].unsqueeze(3)

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

        class_loss = self._get_class_loss(y_trues, y_preds, vectorized_obj_indicator_ij)
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
        # print(f"total_loss: {total_loss}")
        return total_loss


class YOLOv1Loss(nn.Module):
    def __init__(
        self,
        S: int,
        B: int,
        C: int,
        lambda_coord: float = 5,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # self.lambda_coord = 5
        # self.lambda_noobj = 0.5
        # mse = (y_pred - y_true)^2
        self.mse = nn.MSELoss(
            reduction="sum"
        )  # no need sum for individual but needed for last line of computation

    def _initiate_loss(self) -> None:
        # bbox loss
        self.bbox_xy_offset_loss = 0
        self.bbox_wh_loss = 0

        # objectness loss
        self.object_conf_loss = 0
        self.no_object_conf_loss = 0

        # class loss
        self.class_loss = 0

    def forward(self, y_preds: torch.Tensor, y_trues: torch.Tensor):
        # y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
        # y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
        # however y_preds was flattened to (batch_size, S*S*(C + B * 5)) = (batch_size, 1470)
        # so we need to reshape it back to (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)

        self._initiate_loss()
        # with autograd.detect_anomaly():
        y_trues = y_trues.reshape(-1, self.S, self.S, self.C + self.B * 5)
        y_preds = y_preds.reshape(-1, self.S, self.S, self.C + self.B * 5)

        batch_size = y_preds.shape[0]

        # if dataloader has a batch size of 2, then our total loss is the average of the two losses.
        # i.e. total_loss = (loss1 + loss2) / 2 where loss1 is the loss for the first image in the
        # batch and loss2 is the loss for the second image in the batch.

        # to calculate total loss for each image we use the following formula:

        for batch_index in range(batch_size):  # batchsize循环
            # I purposely loop row as inner loop since in python
            # y_ij = y_preds[batch_index, j, i, :]
            for col in range(self.S):  # x方向网格循环
                for row in range(self.S):  # y方向网格循环
                    # this double loop is like matrix: if a matrix
                    # M is of shape (S, S) = (7, 7) then this double loop is
                    # M_ij where i is the row and j is the column.
                    # so first loop is M_{11}, second loop is M_{12}...

                    # check 4 suffice cause by construction both index 4 and 9 will be filled with
                    # the same objectness score (0 or 1)
                    indicator_obj_ij = y_trues[batch_index, row, col, 4] == 1
                    if indicator_obj_ij:
                        # indicator_obj_ij means if has object then calculate else 0
                        b = y_trues[batch_index, row, col, 0:4]
                        bhat_1 = y_preds[batch_index, row, col, 0:4]
                        bhat_2 = y_preds[batch_index, row, col, 5:9]

                        iou_b1 = intersection_over_union(b, bhat_1, bbox_format="yolo")[
                            0
                        ]
                        iou_b2 = intersection_over_union(b, bhat_2, bbox_format="yolo")[
                            0
                        ]

                        x_ij, y_ij, w_ij, h_ij = b

                        if iou_b1 > iou_b2:
                            xhat_ij = bhat_1[..., 0]
                            yhat_ij = bhat_1[..., 1]
                            what_ij = bhat_1[..., 2]
                            hhat_ij = bhat_1[..., 3]
                            # C_ij = max_{bhat \in {bhat_1, bhat_2}} IoU(b, bhat)
                            C_ij = y_trues[batch_index, row, col, 4]  # iou_b1
                            # can be denoted Chat1_ij
                            Chat_ij = y_preds[batch_index, row, col, 4]
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            C_complement_ij = iou_b2
                            Chat_complement_ij = y_preds[batch_index, row, col, 9]
                        else:
                            xhat_ij = bhat_2[..., 0]
                            yhat_ij = bhat_2[..., 1]
                            what_ij = bhat_2[..., 2]
                            hhat_ij = bhat_2[..., 3]
                            C_ij = y_trues[batch_index, row, col, 9]  # iou_b2

                            # can be denoted Chat2_ij
                            Chat_ij = y_preds[batch_index, row, col, 9]
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou1
                            C_complement_ij = iou_b1
                            Chat_complement_ij = y_preds[batch_index, row, col, 4]

                        self.bbox_xy_offset_loss = (
                            self.bbox_xy_offset_loss
                            + self.mse(x_ij, xhat_ij)
                            + self.mse(y_ij, yhat_ij)
                        )

                        # self.bbox_xy_offset_loss = (
                        #     torch.sum((x_ij - xhat_ij) ** 2) ** 2
                        #     + torch.sum((y_ij - yhat_ij) ** 2) ** 2
                        # )

                        # make them abs as sometimes the preds can be negative if no sigmoid layer.
                        # add 1e-6 for stability
                        self.bbox_wh_loss = (
                            self.bbox_wh_loss
                            + self.mse(
                                torch.sqrt(w_ij),
                                torch.sqrt(torch.abs(what_ij + 1e-6)),
                            )
                            + self.mse(
                                torch.sqrt(h_ij),
                                torch.sqrt(torch.abs(hhat_ij + 1e-6)),
                            )
                        )

                        self.object_conf_loss = self.object_conf_loss + self.mse(
                            C_ij, Chat_ij
                        )

                        # obscure as hell no_object_conf_loss...

                        # self.no_object_conf_loss = (
                        #     self.no_object_conf_loss
                        #     + self.mse(C_complement_ij, Chat_complement_ij)
                        # )
                        self.no_object_conf_loss = self.no_object_conf_loss + torch.sum(
                            (0 - Chat_complement_ij) ** 2
                        )

                        self.class_loss = self.class_loss + self.mse(
                            y_trues[batch_index, row, col, 10:],
                            y_preds[batch_index, row, col, 10:],
                        )
                    else:
                        # no_object_conf is constructed to be 0 in ground truth y
                        # can use mse but need to put torch.tensor(0) to gpu
                        # broadcast using 0
                        self.no_object_conf_loss = self.no_object_conf_loss + torch.sum(
                            (0 - y_preds[batch_index, row, col, [4, 9]]) ** 2
                        )

        total_loss = (
            self.lambda_coord * self.bbox_xy_offset_loss
            + self.lambda_coord * self.bbox_wh_loss
            + self.object_conf_loss
            + self.lambda_noobj * self.no_object_conf_loss
            + self.class_loss
        )

        total_loss_averaged_over_batch = total_loss / batch_size
        # print(f"total_loss_averaged_over_batch {total_loss_averaged_over_batch}")

        return total_loss_averaged_over_batch


# # reshape to [49, 30] from [7, 7, 30]
# class YOLOv1Loss2D(nn.Module):
#     # assumptions
#     # 1. B = 2 is a must since we have iou_1 vs iou_2
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
#         # mse = (y_pred - y_true)^2
#         # FIXME: still unclean cause by right reduction is not sum since we are
#         # adding scalars, but class_loss uses vector sum reduction so need to use for all?
#         self.mse = nn.MSELoss(reduction="none")

#     # important step if not the curr batch loss will be added to the next batch loss which causes error.
#     def _initiate_loss(self) -> None:
#         # bbox loss
#         self.bbox_xy_offset_loss = 0
#         self.bbox_wh_loss = 0

#         # objectness loss
#         self.object_conf_loss = 0
#         self.no_object_conf_loss = 0

#         # class loss
#         self.class_loss = 0

#     def compute_xy_offset_loss(self, x_i, xhat_i, y_i, yhat_i) -> torch.Tensor:
#         return self.mse(x_i, xhat_i) + self.mse(y_i, yhat_i)

#     def compute_wh_loss(
#         self, w_i, what_i, h_i, hhat_i, epsilon: float = 1e-6
#     ) -> torch.Tensor:
#         return self.mse(
#             torch.sqrt(w_i), torch.sqrt(torch.abs(what_i + epsilon))
#         ) + self.mse(torch.sqrt(h_i), torch.sqrt(torch.abs(hhat_i + epsilon)))

#     def compute_object_conf_loss(self, conf_i, confhat_i) -> torch.Tensor:
#         return self.mse(conf_i, confhat_i)

#     def compute_no_object_conf_loss(
#         self, conf_complement_i, confhat_complement_i
#     ) -> torch.Tensor:
#         return self.mse(conf_complement_i, confhat_complement_i)

#     def compute_class_loss(self, p_i, phat_i) -> torch.Tensor:
#         # self.mse(p_i, phat_i) is vectorized version
#         total_class_loss = 0
#         for c in range(self.C):
#             total_class_loss += self.mse(p_i[c], phat_i[c])
#         return total_class_loss

#     def forward(self, y_trues: torch.Tensor, y_preds: torch.Tensor):
#         self._initiate_loss()

#         # y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30) -> (batch_size, 49, 30)
#         # y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30) -> (batch_size, 49, 30)
#         y_trues = y_trues.reshape(-1, self.S * self.S, self.C + self.B * 5)
#         y_preds = y_preds.reshape(-1, self.S * self.S, self.C + self.B * 5)

#         batch_size = y_preds.shape[0]

#         # batch_index is the index of the image in the batch
#         for batch_index in range(batch_size):

#             # this is the gt and pred matrix in my notes: y and yhat
#             y_true = y_trues[batch_index]  # (49, 30)
#             y_pred = y_preds[batch_index]  # (49, 30)

#             # i is the grid cell index in my notes ranging from 0 to 48
#             for i in range(self.S * self.S):
#                 # check 4 suffice cause by construction both index 4 and 9 will be filled with
#                 # indicator_obj_i corresponds to 1_i^obj and not 1_ib^obj, this means if has object then calculate else 0
#                 indicator_obj_i = y_true[i, 4] == 1

#                 if indicator_obj_i:

#                     b = y_true[i, 0:4]
#                     bhat_1 = y_pred[i, 0:4]
#                     bhat_2 = y_pred[i, 5:9]

#                     # area of overlap
#                     iou_b1 = intersection_over_union(b, bhat_1, bbox_format="yolo")[0]
#                     iou_b2 = intersection_over_union(b, bhat_2, bbox_format="yolo")[0]

#                     x_i, y_i, w_i, h_i = b

#                     # at this point in time, conf_i is either 0 or 1
#                     conf_i = y_true[i, 4]

#                     p_i = y_true[i, 10:]
#                     phat_i = y_pred[i, 10:]

#                     # max iou
#                     if iou_b1 > iou_b2:
#                         xhat_i = bhat_1[..., 0]
#                         yhat_i = bhat_1[..., 1]
#                         what_i = bhat_1[..., 2]
#                         hhat_i = bhat_1[..., 3]
#                         # conf_i = max_{bhat \in {bhat_1, bhat_2}} IoU(b, bhat)
#                         # by right should be iou_b1 but y_true[i, 4] gives better results as of now
#                         conf_i = y_true[i, 4]
#                         # can be denoted Chat1_i
#                         confhat_i = y_pred[i, 4]
#                         # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
#                         conf_complement_i = iou_b2
#                         confhat_complement_i = y_pred[i, 9]
#                     else:
#                         xhat_i = bhat_2[..., 0]
#                         yhat_i = bhat_2[..., 1]
#                         what_i = bhat_2[..., 2]
#                         hhat_i = bhat_2[..., 3]
#                         # by right should be iou_b2 but y_true[i, 4] gives better results as of now
#                         # note y_true[i, 4] is the same as y_true[i, 9]
#                         conf_i = y_true[i, 9]

#                         # can be denoted Chat2_i
#                         confhat_i = y_pred[i, 9]
#                         # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou1
#                         conf_complement_i = iou_b1
#                         confhat_complement_i = y_pred[i, 4]

#                     # equation 1
#                     self.bbox_xy_offset_loss += self.compute_xy_offset_loss(
#                         x_i, xhat_i, y_i, yhat_i
#                     )

#                     # make them abs as sometimes the preds can be negative if no sigmoid layer.
#                     # add 1e-6 for stability
#                     self.bbox_wh_loss += self.compute_wh_loss(w_i, what_i, h_i, hhat_i)

#                     self.object_conf_loss += self.compute_object_conf_loss(
#                         conf_i, confhat_i
#                     )

#                     # obscure as hell no_object_conf_loss...

#                     # self.no_object_conf_loss = (
#                     #     self.no_object_conf_loss
#                     #     + self.mse(C_complement_ij, Chat_complement_ij)
#                     # )
#                     # FIXME: by right this is not the same as paper but gives better initial results
#                     # FIXME: REMOVED on 25 september 2022 cause I think no need.
#                     self.no_object_conf_loss = self.no_object_conf_loss + torch.sum(
#                         (0 - confhat_complement_i) ** 2
#                     )

#                     self.class_loss += self.compute_class_loss(p_i, phat_i)
#                 else:
#                     # no_object_conf is constructed to be 0 in ground truth y
#                     # can use mse but need to put torch.tensor(0) to gpu
#                     # broadcast using 0
#                     # FIXME: consider unpack 0 from b
#                     # conf_i = 0 for ground truth by definition
#                     # essentially just 0 - confhat_i for 2 bboxes
#                     for j in range(self.B):
#                         self.no_object_conf_loss += self.mse(
#                             y_true[i, 4],
#                             y_pred[i, 4 + j * 5],
#                         )
#             print("\n")
#         total_loss = (
#             self.lambda_coord * self.bbox_xy_offset_loss
#             + self.lambda_coord * self.bbox_wh_loss
#             + self.object_conf_loss
#             + self.lambda_noobj * self.no_object_conf_loss
#             + self.class_loss
#         )
#         # if dataloader has a batch size of 2, then our total loss is the average of the two losses.
#         # i.e. total_loss = (loss1 + loss2) / 2 where loss1 is the loss for the first image in the
#         # batch and loss2 is the loss for the second image in the batch.
#         total_loss_averaged_over_batch = total_loss / batch_size
#         # print(f"total_loss_averaged_over_batch {total_loss_averaged_over_batch}")

#         return total_loss_averaged_over_batch


# reshape to [49, 30] from [7, 7, 30]
class YOLOv1Loss2D(nn.Module):
    # assumptions
    # 1. B = 2 is a must since we have iou_1 vs iou_2
    def __init__(
        self,
        S: int,
        B: int,
        C: int,
        lambda_coord: float = 5,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # mse = (y_pred - y_true)^2
        # FIXME: still unclean cause by right reduction is not sum since we are
        # adding scalars, but class_loss uses vector sum reduction so need to use for all?
        self.mse = nn.MSELoss(reduction="none")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # important step if not the curr batch loss will be added to the next batch loss which causes error.
    def _initiate_loss(self) -> None:
        # bbox loss
        self.bbox_xy_offset_loss = 0
        self.bbox_wh_loss = 0

        # objectness loss
        self.object_conf_loss = 0
        self.no_object_conf_loss = 0

        # class loss
        self.class_loss = 0

    def compute_xy_offset_loss(self, x_i, xhat_i, y_i, yhat_i) -> torch.Tensor:
        return self.mse(x_i, xhat_i) + self.mse(y_i, yhat_i)

    def compute_wh_loss(
        self, w_i, what_i, h_i, hhat_i, epsilon: float = 1e-6
    ) -> torch.Tensor:
        return self.mse(
            torch.sqrt(w_i), torch.sqrt(torch.abs(what_i + epsilon))
        ) + self.mse(torch.sqrt(h_i), torch.sqrt(torch.abs(hhat_i + epsilon)))

    def compute_object_conf_loss(self, conf_i, confhat_i) -> torch.Tensor:
        return self.mse(conf_i, confhat_i)

    def compute_no_object_conf_loss(self, conf_i, confhat_i) -> torch.Tensor:
        return self.mse(conf_i, confhat_i)

    def compute_class_loss(self, p_i, phat_i) -> torch.Tensor:
        # self.mse(p_i, phat_i) is vectorized version
        total_class_loss = 0
        for c in range(self.C):
            total_class_loss += self.mse(p_i[c], phat_i[c])
        return total_class_loss

    def forward(self, y_trues: torch.Tensor, y_preds: torch.Tensor):
        self._initiate_loss()

        # y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30) -> (batch_size, 49, 30)
        # y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30) -> (batch_size, 49, 30)
        y_trues = y_trues.reshape(-1, self.S * self.S, self.C + self.B * 5)
        y_preds = y_preds.reshape(-1, self.S * self.S, self.C + self.B * 5)

        batch_size = y_preds.shape[0]

        # batch_index is the index of the image in the batch
        for batch_index in range(batch_size):
            # print(f"batch_index {batch_index}")

            # this is the gt and pred matrix in my notes: y and yhat
            y_true = y_trues[batch_index]  # (49, 30)
            y_pred = y_preds[batch_index]  # (49, 30)
            # print(bmatrix(y_true.cpu().detach().numpy()))
            # print(bmatrix(y_pred.cpu().detach().numpy()))
            
            # print(f"y_true {y_true}")

            # i is the grid cell index in my notes ranging from 0 to 48
            for i in range(self.S * self.S):
                # print(f"row/grid cell index {i}")

                y_true_i = y_true[i]  # (30,) or (1, 30)
                y_pred_i = y_pred[i]  # (30,) or (1, 30)

                # print(f"y_i={y_true_i}")
                # print(f"yhat_i={y_pred_i}")

                # b = y_true_i[0:4]

                # this is $\obji$ and checking y_true[i, 4] is sufficient since y_true[i, 9] is repeated
                indicator_obj_i = y_true_i[4] == 1

                # here equation 1, 2, 3, 5
                if indicator_obj_i:

                    b = y_true_i[0:4]
                    bhat_1 = y_pred_i[0:4]
                    bhat_2 = y_pred_i[5:9]
                    # print(f"b {b}\nbhat_1 {bhat_1}\nbhat_2 {bhat_2}")

                    x_i, y_i, w_i, h_i = b
                    # print(f"x_i {x_i}, y_i {y_i}, w_i {w_i}, h_i {h_i}")

                    p_i = y_true_i[10:]
                    phat_i = y_pred_i[10:]

                    # area of overlap
                    iou_b1 = intersection_over_union(b, bhat_1, bbox_format="yolo")
                    iou_b2 = intersection_over_union(b, bhat_2, bbox_format="yolo")

                    # max iou
                    if iou_b1 > iou_b2:
                        xhat_i, yhat_i, what_i, hhat_i = bhat_1
                        # conf_i = max_{bhat \in {bhat_1, bhat_2}} IoU(b, bhat)
                        # however I set conf_i = y_true_i[4] = 1 here as it gives better results for our case
                        conf_i, confhat_i = y_true_i[4], y_pred_i[4]
                        
                        # fmt: off
                        # equation 1
                        self.bbox_xy_offset_loss += self.lambda_coord * self.compute_xy_offset_loss(x_i, xhat_i, y_i, yhat_i)

                        # make them abs as sometimes the preds can be negative if no sigmoid layer.
                        # add 1e-6 for stability
                        self.bbox_wh_loss += self.lambda_coord * self.compute_wh_loss(w_i, what_i, h_i, hhat_i)

                        self.object_conf_loss += self.compute_object_conf_loss(conf_i, confhat_i)

                        # mention 2 other ways
                        # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                        # we can set conf_i = iou_b2 as well as the smaller of the two should be optimized to say there exist no object instead of proposing something.
                        # we can set conf_i = 0 as well and it will work.
                        self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=torch.tensor(0., device="cuda"), confhat_i=y_pred_i[9])

                    else:
                        xhat_i, yhat_i, what_i, hhat_i = bhat_2
                        # same as above
                        conf_i, confhat_i = y_true_i[4], y_pred_i[9]

                        # equation 1
                        self.bbox_xy_offset_loss += self.lambda_coord * self.compute_xy_offset_loss(x_i, xhat_i, y_i, yhat_i)

                        # make them abs as sometimes the preds can be negative if no sigmoid layer.
                        # add 1e-6 for stability
                        self.bbox_wh_loss += self.lambda_coord * self.compute_wh_loss(w_i, what_i, h_i, hhat_i)

                        self.object_conf_loss += self.compute_object_conf_loss(conf_i, confhat_i)

                        # mention 2 other ways
                        # we can set conf_i = iou_b1 as well as the smaller of the two should be optimized to say there exist no object instead of proposing something.
                        # we can set conf_i = 0 as well and it will work.
                        self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=torch.tensor(0., device=self.device), confhat_i=y_pred_i[4])

                    self.class_loss += self.compute_class_loss(p_i, phat_i)
                else:
                    # equation 4
                    # recall if there is no object, then conf_i is 0 by definition i.e. y_true_i[4] = 0
                    # confhat_i is y_pred_i[4] and y_pred_i[9] 
                    # it is worth noting that we loop over both bhat_1 and bhat_2 and calculate
                    # the no object loss for both of them. This is because we want to penalize
                    # the model for predicting an object when there is no object **for both bhat_1 and bhat_2**.
                    # in paper it seems that they only penalize for bhat_i^{\jmax} since they used 1_{i}^{noobj} notation
                    for j in range(self.B):
                        self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=y_true_i[4], confhat_i=y_pred_i[4 + j * 5])


        total_loss = (
            self.bbox_xy_offset_loss
            + self.bbox_wh_loss
            + self.object_conf_loss
            + self.no_object_conf_loss
            + self.class_loss
        )

        # if dataloader has a batch size of 2, then our total loss is the average of the two losses.
        # i.e. total_loss = (loss1 + loss2) / 2 where loss1 is the loss for the first image in the
        # batch and loss2 is the loss for the second image in the batch.
        total_loss_averaged_over_batch = total_loss / batch_size
        # print(f"total_loss_averaged_over_batch {total_loss_averaged_over_batch}")

        return total_loss_averaged_over_batch

# # reshape to [49, 30] from [7, 7, 30]
# # incorporate bbox index in the B=2 dimension
# class YOLOv1Loss2Dv2(nn.Module):
#     # assumptions
#     # 1. B = 2 is a must since we have iou_1 vs iou_2
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

#         # mse = (y_pred - y_true)^2
#         # FIXME: still unclean cause by right reduction is not sum since we are
#         # adding scalars, but class_loss uses vector sum reduction so need to use for all?
#         self.mse = nn.MSELoss(reduction="sum")

#     # important step if not the curr batch loss will be added to the next batch loss which causes error.
#     def _initiate_loss(self) -> None:
#         # bbox loss
#         self.bbox_xy_offset_loss = 0
#         self.bbox_wh_loss = 0

#         # objectness loss
#         self.object_conf_loss = 0
#         self.no_object_conf_loss = 0

#         # class loss
#         self.class_loss = 0

#     def forward(self, y_preds: torch.Tensor, y_trues: torch.Tensor):
#         self._initiate_loss()

#         # y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
#         # y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30)
#         # reshape all to (batch_size, 49, 30) for easier computation
#         # shape: (batch_size, 7*7, 30) = (batch_size, 49, 30)
#         y_trues = y_trues.reshape(-1, self.S * self.S, self.C + self.B * 5)
#         y_preds = y_preds.reshape(-1, self.S * self.S, self.C + self.B * 5)

#         batch_size = y_preds.shape[0]

#         # if dataloader has a batch size of 2, then our total loss is the average of the two losses.
#         # i.e. total_loss = (loss1 + loss2) / 2 where loss1 is the loss for the first image in the
#         # batch and loss2 is the loss for the second image in the batch.
#         for batch_index in range(
#             batch_size
#         ):  # batch_index is the index of the image in the batch

#             # subset here to stay consistent with my notes instead of doing it
#             # more obscurely as y_trues[batch_index, grid_cell_index, ...] later
#             # now y_true and y_pred refers to 1 single image **important to know**
#             y_true = y_trues[batch_index]  # (49, 30)
#             y_pred = y_preds[batch_index]  # (49, 30)

#             # grid_cell_index ranges from 0 to 48 and goes from top left to bottom right
#             for i in range(self.S * self.S):  # i = grid_cell_index
#                 b = y_true[i, 0:4]  # b_true
#                 iou_bhat_j_max = iou_bhat_j
#                 j_max = j

#                 for j in range(self.B):  # j = bbox_index
#                     # j=0 -> 0:4, j=1 -> 5:9...
#                     bhat_j = y_pred[i, j * 5 : j * 5 + 4]
#                     iou_bhat_j = intersection_over_union(b, bhat_j, bbox_format="yolo")[
#                         0
#                     ]

#                     # find the bbox with the highest iou
#                     if iou_bhat_j > iou_bhat_j_max:
#                         iou_bhat_j_max = iou_bhat_j
#                         j_max = j

#                 # now we have the bbox with the highest iou
#                 for j in range(self.B):
#                     # j=0 -> 0:4, j=1 -> 5:9...
#                     bhat_j = y_pred[i, j * 5 : j * 5 + 4]

#                     x_ij, y_ij, w_ij, h_ij = b
#                     xhat_ij, yhat_ij, what_ij, hhat_ij = (
#                         bhat_j[..., 0],
#                         bhat_j[..., 1],
#                         bhat_j[..., 2],
#                         bhat_j[..., 3],
#                     )

#                     conf_ij = y_true[i, 4]
#                     confhat_ij = bhat_j[..., 4]

#                     if j == j_max:  # this is checking our 1_ij^obj
#                         # then compute
#                         # 1. x and y offset loss
#                         self.bbox_xy_offset_loss = (
#                             self.bbox_xy_offset_loss
#                             + self.mse(x_ij, xhat_ij)
#                             + self.mse(y_ij, yhat_ij)
#                         )

#                         # 2. width and height loss
#                         self.bbox_wh_loss = (
#                             self.bbox_wh_loss
#                             + self.mse(
#                                 torch.sqrt(w_ij),
#                                 torch.sqrt(torch.abs(what_ij + 1e-6)),
#                             )
#                             + self.mse(
#                                 torch.sqrt(h_ij),
#                                 torch.sqrt(torch.abs(hhat_ij + 1e-6)),
#                             )
#                         )

#                         # 3. objectness loss
#                         self.object_conf_loss = self.object_conf_loss + self.mse(
#                             conf_ij, confhat_ij
#                         )

#                         # 4. no objectness loss
#                         self.no_object_conf_loss = self.no_object_conf_loss + torch.sum(
#                             (0 - confhat_ij) ** 2
#                         )

#                     else:
#                         # purposely put pass to indicate they all are zero
#                         pass


if __name__ == "__main__":
    torch.set_printoptions(threshold=5000)
    batch_size = 4
    S, B, C = 7, 2, 20
    lambda_coord, lambda_noobj = 5, 0.5

    # load directly the first batch of the train loader
    y_trues = torch.load("y_trues.pt")
    y_preds = torch.load("y_preds.pt")

    # print(y_trues.shape)
    # print(y_trues[0][3,3,:])

    criterion = YOLOv1Loss2D(S, B, C, lambda_coord, lambda_noobj)
    loss = criterion(y_trues, y_preds)