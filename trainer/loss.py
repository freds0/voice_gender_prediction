import torch.nn.functional as F
import torch

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

