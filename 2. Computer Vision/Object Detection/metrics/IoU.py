import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # boxes_preds shape: (N, 4) -> N: number of bounding boxes
    # boxes_labels shape: (N, 4) 

    # midpoint (xywh) -> make "corners" box format (xyxy)
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # corners (xyxy)
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3] # ... 은 이전의 모든 dimension은 그대로 유지한다는 의미
        box1_y2 = boxes_preds[..., 3:4] # output tensor shape을 (N, 1)로 만들기 위해서 slicing [i, i+1]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # (x1, y1): top-left of intersection area
    # (x2, y2): bottom-right of intersection area
    x1 = torch.max(box1_x1, box2_x1, dim=-1)
    x2 = torch.max(box1_x2, box2_x2, dim=-1)
    y1 = torch.min(box1_y1, box2_y1, dim=-1)
    y2 = torch.min(box1_y2, box2_y2, dim=-1) # shape (N)

    # .clamp(0) is for the case when they do not intersect
    # torch.clamp(input, min=None, max=None, *, out=None) → Tensor
    #   -> minimum bound와 upper bound를 정해서 해당 값을 벗어나면 min/max로 치환
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6) # 1e-6 for 나누기 연산
    