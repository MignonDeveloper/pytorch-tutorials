import torch

from iou import intersection_over_union

def non_max_suppression(
    predictions,
    iou_threshold,
    prob_threshold,
    box_format="corners"
):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    # predictions examples = [[1, 0.9, x1, y1, x2, y2]]
    # class, probability score, coordinates

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold] # 최소 prob를 넘어가는 box만 최종 output을 위해서 사용
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # prob_score를 기준으로 모든 box들을 오름차순으로 정렬한다.
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes 
            if box[0] != chosen_box[0] # class가 다른 경우는 그대로 유지
            or intersection_over_union( # 같은 class에 대해서 IoU값이 threshold를 넘어가면 box 삭제
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])),
                box_format=box_format
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box) # 살아남은 box를 하나씩 계속 추가
    
    return bboxes_after_nms