import torch
import torch.nn.functional as F

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for batch_idx, (data, target, _) in enumerate(dataset):

        data, true_mask = data.cuda().float(), target.cuda().long()

        mask_pred = net(data)
        mask_pred = (mask_pred > 0.5).float()
        mask_pred = mask_pred.argmax(dim=1)
        tot += dice_coeff(mask_pred.float(), true_mask.float()).item()
    return tot / (batch_idx + 1)
