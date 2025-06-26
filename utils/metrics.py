import torch

def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) >= 1).float().sum(dim=(1, 2, 3))
    iou = intersection / (union + 1e-6)
    return iou.mean().item()

def compute_precision_recall_f1(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    TP = (pred * target).sum(dim=(1, 2, 3))
    FP = (pred * (1 - target)).sum(dim=(1, 2, 3))
    FN = ((1 - pred) * target).sum(dim=(1, 2, 3))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision.mean().item(), recall.mean().item(), f1.mean().item()
