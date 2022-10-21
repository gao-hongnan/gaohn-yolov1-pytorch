"""
Implementation of Yolo Loss Function from the original yolo paper
"""
import torch
from torch import nn
from utils import intersection_over_union, bmatrix
import numpy as np
import pandas as pd

# reshape to [49, 30] from [7, 7, 30]
class YOLOv1Loss2D(nn.Module):
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

    def _initiate_loss(self) -> None:
        """Initializes all the loss values to 0.
        
        Note:
            This is an important step if not the current batch loss will be added to the
            next batch loss which causes error in `loss.backward()`.
        """
        # bbox loss
        self.bbox_xy_offset_loss = 0
        self.bbox_wh_loss = 0

        # objectness loss
        self.object_conf_loss = 0
        self.no_object_conf_loss = 0

        # class loss
        self.class_loss = 0

    def compute_xy_offset_loss(
        self,
        x_i: torch.Tensor,
        xhat_i: torch.Tensor,
        y_i: torch.Tensor,
        yhat_i: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the loss for the x and y offset of the bounding box."""
        x_offset_loss = self.mse(x_i, xhat_i)
        #print("offsetxy",x_offset_loss)
        y_offset_loss = self.mse(y_i, yhat_i)
        xy_offset_loss = x_offset_loss + y_offset_loss
        return xy_offset_loss

    def compute_wh_loss(
        self,
        w_i: torch.Tensor,
        what_i: torch.Tensor,
        h_i: torch.Tensor,
        hhat_i: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """Computes the loss for the width and height of the bounding box.
        
        Note:
            The width and height are predicted as the square root of the width and height
            and absolute values are applied to the predictions as they are unbounded
            below and can be numerically unstable.
        """
        w_loss = self.mse(torch.sqrt(w_i), torch.sqrt(torch.abs(what_i + epsilon)))
        h_loss = self.mse(torch.sqrt(h_i), torch.sqrt(torch.abs(hhat_i + epsilon)))
        wh_loss = w_loss + h_loss
        return wh_loss

    def compute_object_conf_loss(
        self, conf_i: torch.Tensor, confhat_i: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss for the object confidence when there is really an object."""
        return self.mse(conf_i, confhat_i)

    def compute_no_object_conf_loss(
        self, conf_i: torch.Tensor, confhat_i: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss for the object confidence when there is no object."""
        return self.mse(conf_i, confhat_i)

    def compute_class_loss(self, p_i: torch.Tensor, phat_i: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the class prediction.

        Note:
            Instead of looping C number of classes, we can use self.mse(p_i, phat_i)
            as the vectorized version.
        """
        total_class_loss = 0
        for c in range(self.C):
            total_class_loss += self.mse(p_i[c], phat_i[c])
        return total_class_loss

    # fmt: off
    def forward(self, y_trues: torch.Tensor, y_preds: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function.

        Args:
            y_trues (torch.Tensor): The ground truth tensor of shape (bs, S, S, 5B + C).
            y_preds (torch.Tensor): The predicted tensor of shape (bs, S, S, 5B + C).

        Returns:
            total_loss_averaged_over_batch (torch.Tensor): The total loss averaged over the batch.
        """
        self._initiate_loss()                                                                       # reset loss values
        
        y_trues = y_trues.reshape(-1, self.S * self.S, self.C + self.B * 5)                         # (4, 49, 30)
        y_preds = y_preds.reshape(-1, self.S * self.S, self.C + self.B * 5)                         # (4, 49, 30)

        batch_size = y_preds.shape[0]                                                               # 4

        for batch_index in range(batch_size):                                                       # for each image in batch 
            y_true = y_trues[batch_index]                                                           # y:    (49, 30)
            y_pred = y_preds[batch_index]                                                           # yhat: (49, 30)

            for i in range(self.S * self.S):                                                        # i is the grid cell index [0, 48] 
                y_true_i = y_true[i]                                                                # y_i:    (30,) or (1, 30)
                y_pred_i = y_pred[i]                                                                # yhat_i: (30,) or (1, 30)
                
                indicator_obj_i = y_true_i[4] == 1                                                  # this is $\obji$ and checking y_true[i, 4] is sufficient since y_true[i, 9] is repeated

                if indicator_obj_i:                                                                 # here equation (a), (b), (c) and (e) of the loss equation on a single grid cell.
                    b = y_true_i[0:4]                                                               # b:    (4,) or (1, 4)
                    bhat_1 = y_pred_i[0:4]                                                          # bhat1: (4,) or (1, 4)
                    bhat_2 = y_pred_i[5:9]                                                          # bhat2: (4,) or (1, 4)
                    
                    x_i, y_i, w_i, h_i = b                                                          # x_i, y_i, w_i, h_i: (1,)
                    # at this stage jmax is not known yet.
                    xhat_i1, yhat_i1, what_i1, hhat_i1 = bhat_1                                     # xhat_i1, yhat_i1, what_i1, hhat_i1: (1,)
                    xhat_i2, yhat_i2, what_i2, hhat_i2 = bhat_2                                     # xhat_i2, yhat_i2, what_i2, hhat_i2: (1,)
                    
                    conf_i, confhat_i1, confhat_i2 = y_true_i[4], y_pred_i[4], y_pred_i[9]          # conf_i, confhat_i1, confhat_i2: (1,)
                    
                    p_i, phat_i = y_true_i[10:], y_pred_i[10:]                                      # p_i, phat_i: (20,) or (1, 20)

                    # area of overlap
                    iou_b1 = intersection_over_union(b, bhat_1, bbox_format="yolo")                 # iou of b and bhat1
                    iou_b2 = intersection_over_union(b, bhat_2, bbox_format="yolo")                 # iou of b and bhat2

                    if iou_b1 > iou_b2:
                        # conf_i = max_{bhat \in {bhat_1, bhat_2}} IoU(b, bhat)
                        # however I set conf_i = y_true_i[4] = 1 here as it gives better results for our case
                        xhat_i_jmax, yhat_i_jmax, what_i_jmax, hhat_i_jmax, confhat_i_jmax = xhat_i1, yhat_i1, what_i1, hhat_i1, confhat_i1
                        confhat_i_complement = confhat_i2
                    else:
                        xhat_i_jmax, yhat_i_jmax, what_i_jmax, hhat_i_jmax, confhat_i_jmax = xhat_i2, yhat_i2, what_i2, hhat_i2, confhat_i2
                        confhat_i_complement = confhat_i1
                        

                    if batch_index == 0 and i == 30:
                        print(batch_index, i)
                        print(f"y_true_i: {y_true_i}")
                        print(f"y_pred_i: {y_pred_i}")
                        print(f"x_i: {x_i}, y_i: {y_i}, w_i: {w_i}, h_i: {h_i}")
                        print(f"xhat_jmax: {xhat_i_jmax}, yhat_jmax: {yhat_i_jmax}, what_jmax: {what_i_jmax}, hhat_jmax: {hhat_i_jmax}, confhat_jmax: {confhat_i_jmax}")
                        
                        print(f"self.bbox_xy_offset_loss: {self.lambda_coord * self.compute_xy_offset_loss(x_i, xhat_i_jmax, y_i, yhat_i_jmax)}")
                        print(f"self.bbox_wh_loss: {self.lambda_coord * self.compute_wh_loss(w_i, what_i_jmax, h_i, hhat_i_jmax)}")
                        print(f"self.object_conf_loss: {self.compute_object_conf_loss(conf_i, confhat_i_jmax)}")
                        # print(f"self.no_object_conf_loss: {self.no_object_conf_loss}")
                        print(f"self.class_loss: {self.compute_class_loss(p_i, phat_i)}")
                    
                    self.bbox_xy_offset_loss += self.lambda_coord * self.compute_xy_offset_loss(x_i, xhat_i_jmax, y_i, yhat_i_jmax)                 # equation 1
                    self.bbox_wh_loss += self.lambda_coord * self.compute_wh_loss(w_i, what_i_jmax, h_i, hhat_i_jmax)                               # equation 2
                    self.object_conf_loss += self.compute_object_conf_loss(conf_i, confhat_i_jmax)                                                  # equation 3

                    # mention 2 other ways
                    # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                    # we can set conf_i = iou_b2 as well as the smaller of the two should be optimized to say there exist no object instead of proposing something.
                    # we can set conf_i = 0 as well and it will work.
                    # TODO: comment for blog if uncomment it performs a bit better early.
                    # self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=torch.tensor(0., device="cuda"), confhat_i=confhat_i_complement)
                    self.class_loss += self.compute_class_loss(p_i, phat_i)                                                                               # equation 5
                else:
                    for j in range(self.B):
                        self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=y_true_i[4], confhat_i=y_pred_i[4 + j * 5]) # equation 4
                    

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

if __name__ == "__main__":
    torch.set_printoptions(threshold=5000)
    np.set_printoptions(threshold=5000)
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

