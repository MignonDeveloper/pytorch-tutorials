import torch
from collections import counter
from iou import intersection_over_union

# single IoU Threshold에 대해서 진행
# 만약 다양한 값이 필요하다면 함수를 여러번 호출하면 끝! -> 만약 효율성을 생각한다면 threshold에 array를 받는 것도 하나의 방법
def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="corners",
    num_classes=20
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


    # pred_boxes, true_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6 # for numerical stability

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in ture_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # example) in class 0: img 0 has 3 bboxes, img 1 has 5 bboxes in ground_truths
        #          amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # amount_boxes = {0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0])}
        # 각 class별로 ground truths의 box 개수에 맞춰서 값을 생성

        detections.sort(key=lambda x: x[2], reverse=True) # prediction score에 맞춰서 내림차순으로 정렬
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections): # 각 class별로 모든 detection에 대해서 탐구 시작
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0] # 같은 img안에 있는 ground truth와 비교를 위해 idx가 같은 이미지의 bbox를 선별
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img): # detection에서 나온 특정 bbox와 img에 존재하는 모든 ground truth bbox와 비교해서 가장 iou가 높은 bbox를 고른다.
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold: # detection과 gt와의 best_iou가 threshold보다 높고
                if amount_bboxes[detection[0]][best_gt_idx] == 0: # 이전에 한번도 check 되지 않은 ground_truth box라면
                    TP[detection_idx] = 1 # TP으로 추가
                    amount_bboxes[detection[0]][best_gt_idx] = 1 # 체크된 ground_truth를 두번확인 하지 않도록~
                else:
                    FP[detection_idx] = 1 # 이미 해당 ground_truth box가 체크되었다면 FP로 추가

            else: # iou threshold를 넘기지 못했다면 FP로 추가
                FP[detection_idx] = 1

        # for calculate precision
        # example) [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon) # element wise
        precision = TP_cumsum / (TP_cumsum + FP_cumsum _ epsilon) # element wise

        precision = torch.cat((torch.tensor([1]), precisions)) # auc 계산을 위해 최종값을 지정
        recalls = torch.cat((torch.tensor([0]), recalls)) # auc 계산을 위해 최종값을 지정
        average_precisions.append(torch.trapz(precisions, recalls)) # area under the precision / recall graph -> trapz(y, x): numerical integration

    
    return sum(average_precisions) / len(average_precisions)

