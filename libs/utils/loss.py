import math
import torch

from .utility import mask_iou

def binary_entropy_loss(pred, target, num_object, eps=0.001):

    ce = - 1.0 * target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)

    loss = torch.mean(ce)

    # TODO: training with bootstrapping

    return loss

def cross_entropy_loss(pred, mask, num_object, bootstrap=0.4):

    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, _, H, W = mask.shape

    pred = -1 * torch.log(pred)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)

    # bootstrap
    num = int(H * W * bootstrap)

    loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1], dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def mask_iou_loss(pred, mask, num_object):

    N, K, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)
    start = 0 if K == num_object else 1

    for i in range(N):
        loss += (1.0 - mask_iou(pred[i, start:num_object+start], mask[i, start:num_object+start]))

    loss = loss / N
    return loss

def obj_iou(pred, target, eps=1e-7):

    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = (inter + eps) / (union + eps)
    # iou = iou.detach()

    return iou

def iou_pred_loss(pred, mask, iou_pred, num_object):
    """
    param: pred
    param: mask
    param: iou_pred
    param: num_object
    """
    N, K, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)
    start = 0 if K == num_object else 1

    for i in range(N):
        loss += 1/(2*num_object)*torch.sum((iou_pred[i, 0:num_object] - obj_iou(pred[i, start:num_object+start], mask[i, start:num_object+start]))**2)

    loss = loss / N
    return loss





