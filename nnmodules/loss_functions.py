import torch
from utils.ops import check_for_nan_inf

def mc_npair_loss(anchors, positives, temperature):
    """N-pair-mc loss: https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf
    
    sum_i log(1 + sum_{j!=i} exp(i*j+ - i*i+))

    Args:
        anchors (TYPE): Description
        positives (TYPE): Description
        temperature (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    # ij+
    # {D x ... x D} x B x B
    outterproduct = torch.matmul(anchors, positives.transpose(-2,-1))

    # ii+
    # {D x ... x D} x B x 1
    innerproduct = torch.sum(anchors*positives, dim=-1, keepdim=True)

    # ij+ - ii+
    # {D x ... x D} x B x B
    difference = (outterproduct - innerproduct)

    # exp(ij+ - ii+)
    # zero-out diagnol entries (should be 0... just a precaution)
    # {D x ... x D} x B x B
    diagnol_zeros = (1-torch.diag(torch.ones(outterproduct.shape[-1]))).unsqueeze(0).to(anchors.device)
    exp_dif = torch.exp(difference/temperature)*diagnol_zeros

    # final loss
    # sum_i log(1 + sum_{j!=i} exp(i*j+ - i*i+))
    # {D x ... x D} x B
    losses_log = (1 + exp_dif.sum(-1)).log()

    # {D x ... x D}
    losses = losses_log.mean(-1)

    check_for_nan_inf(losses)

    # ======================================================
    # stats
    # ======================================================
    anchor_negative = outterproduct*diagnol_zeros
    B = anchor_negative.shape[1]
    correct = (innerproduct >= anchor_negative).sum(-1).flatten(0,1) == B

    anchor_negative_mean = anchor_negative[anchor_negative != 0].mean().item()
    anchor_positive_mean=innerproduct.mean().item()
    positive_negative_difference = anchor_positive_mean - anchor_negative_mean
    stats=dict(
        anchor_negative=anchor_negative_mean,
        anchor_positive=anchor_positive_mean,
        positive_negative_difference=positive_negative_difference,
        accuracy=correct.float().mean().item(),
        )
    return losses, stats