import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.labmda_coord = 5
    
    def forward(self, predictions, target): # target도 각 grid cell마다 모두 labeling을 해준 형태인가보네!
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, best_box = torch.max(ious, dim=0) # out (tuple, optional) – the result tuple of two output tensors (max, max_indices)
        exists_box = target[..., 20].unsqueeze(3) # identity_obj_i


        # ======================= #
        #   FOR BOX COORDINATES   #
        # ======================= #
        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25]) # iou가 더 높은 box를 box_prediction으로 결정
        )
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * # gradient를 올바르게 보내주기 위해서 해당 숫자의 부호를 기억
                                    torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)) # 음수값이 나올 수도 있기 때문에! abs로 처리
        box_targets = exists_box * target[..., 21:25]
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2), # 마지막 4(x,y,w,h)는 유지
            torch.flatten(box_targets, end_dim=-2)
        )


        # ======================= #
        #     FOR OBJECT LOSS     #
        # ======================= #
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[... , 20: 21]
        )

        # (N, S, S) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box, s),
            torch.flatten(exists_box * target[..., 20:21])
        )


        # ======================= #
        #    FOR NO OBJECT LOSS   #
        # ======================= #
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )


        # ======================= #
        #      FOR CLASS LOSS     #
        # ======================= #
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_object_loss +
            self.class_loss
        )

        return loss

