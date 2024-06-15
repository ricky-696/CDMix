import torch
from typing import Tuple


def mask_to_bbox(mask: torch.tensor):
    """
        Convert mask to bounding box
        input: mask (Tensor[H, W])
    """
    if mask.dim() == 3 and mask.size(0) == 1:
        mask = mask.squeeze(0)
    else:
        assert mask.dim() == 2, "mask must be 2D tensor"

    non_zero_indices = torch.nonzero(mask, as_tuple=False)
    
    if non_zero_indices.numel() > 0:
        min_x = non_zero_indices[:, 1].min().item()
        min_y = non_zero_indices[:, 0].min().item()
        max_x = non_zero_indices[:, 1].max().item()
        max_y = non_zero_indices[:, 0].max().item()
        
        return torch.tensor([[min_x, min_y, max_x, max_y]]) # Tensor[N, 4]
    
    return torch.tensor(0)


def _loss_inter_union(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

    return intsctk, unionk

def diou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    assert (boxes1[:, 2] >= boxes1[:, 0]).all(), "x2 must be greater than or equal to x1 in boxes1"
    assert (boxes1[:, 3] >= boxes1[:, 1]).all(), "y2 must be greater than or equal to y1 in boxes1"
    assert (boxes2[:, 2] >= boxes2[:, 0]).all(), "x2 must be greater than or equal to x1 in boxes2"
    assert (boxes2[:, 3] >= boxes2[:, 1]).all(), "y2 must be greater than or equal to y1 in boxes2"

    intsct, union = _loss_inter_union(boxes1, boxes2)
    iou = intsct / (union + eps)

    # smallest enclosing box
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    # The diagonal distance of the smallest enclosing box squared
    diagonal_distance_squared = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    # The distance between boxes' centers squared.
    centers_distance_squared = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared)

    return loss, iou


if __name__ == '__main__':
    boxes1 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)

    loss, iou = diou_loss(boxes1, boxes2)
    print("DIoU Loss:", loss.item())
    print("IoU:", iou.item())
