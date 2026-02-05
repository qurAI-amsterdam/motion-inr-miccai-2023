import torch
import numpy as np
from scipy.ndimage.measurements import label as scipy_label


def getLargestCC(segmentation, ndim=3):
    assert ndim in [2, 3]
    if ndim == 3:
        struc = np.ones((3, 3, 3))
    else:
        struc = np.ones((3, 3))
    labels, count = scipy_label(segmentation, structure=struc)
    assert labels.max() != 0  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def compute_overlap(mask1, mask2, classes=(1, 2, 3)):
    losses = []
    for c in classes:
        bin_m1, bin_m2 = (mask1 == c).astype(np.int32), (mask2 == c).astype(np.int32)
        nominator = np.sum(bin_m1 * bin_m2)
        denominator = np.sum(bin_m1) + np.sum(bin_m2)
        if denominator > 0:
            losses.append(2 * nominator / denominator)
        else:
            losses.append(0)
    return np.asarray(losses)


def soft_dice_score(prob_c, one_hot, reduction="mean"):
    """

    Computing the soft-dice-loss for a SPECIFIC class according to:

    DICE_c = \frac{\sum_{i=1}^{N} (R_c(i) * A_c(i) ) }{ \sum_{i=1}^{N} (R_c(i) +   \sum_{i=1}^{N} A_c(i)  }

    Input: (1) probs: 4-dim tensor [batch_size, num_of_classes, width, height]
               contains the probabilities for each class
           (2) true_binary_labels, 4-dim tensor with the same dimensionalities as probs, but contains binary
           labels for a specific class

    """
    eps = 1.0e-6
    nominator = 2 * torch.sum(one_hot * prob_c, dim=(2, 3))
    denominator = torch.sum(one_hot, dim=(2, 3)) + torch.sum(prob_c, dim=(2, 3)) + eps
    if reduction == "mean":
        return -torch.mean(nominator / denominator)
    else:
        return -nominator / denominator


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target, is_one_hot=False):
        # we expect input/target is [b, 1, y, x]
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).long()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).long()
        if not is_one_hot:
            one_hot_shape = (
                1,
                self.n_classes,
            ) + input.shape[-2:]
            target = (
                torch.zeros(one_hot_shape)
                .to(input.device)
                .scatter_(1, target.unsqueeze(1), 1)
            )
        return soft_dice_score(input, target)


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, eps=1.0e-6):
        # we expect input/target is [b, y, x]
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        assert pred.dim() == 3
        assert target.dim() == 3
        nominator = 2 * torch.sum(target * pred, dim=(1, 2))
        denominator = torch.sum(target, dim=(1, 2)) + torch.sum(pred, dim=(1, 2)) + eps
        if self.reduction == "mean":
            return -torch.mean(nominator / denominator)
        else:
            return -nominator / denominator


def intersection_union(pred, target, is_one_hot=False):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).long()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()

    if not is_one_hot:
        target = (
            torch.zeros(pred.shape).to(pred.device).scatter_(1, target.unsqueeze(1), 1)
        )
    smooth = 1.0

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))
