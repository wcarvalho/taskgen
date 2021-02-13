import torch
from sfgen.tools.ops import check_for_nan_inf

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
    # ... x B/2 x B/2
    # ij+
    outterproduct = torch.matmul(anchors, positives.transpose(-2,-1))

    # ... x B/2 x 1
    # ii+
    innerproduct = torch.sum(anchors*positives, dim=-1, keepdim=True)

    # ij+ - ii+
    # where zero-out diagnol entries (should be 0... just a precaution)
    diagnol_zeros = (1-torch.diag(torch.ones(outterproduct.shape[-1]))).unsqueeze(0).to(anchors.device)
    difference = (outterproduct - innerproduct)*diagnol_zeros

    # exp(ij+ - ii+)
    exp_dif = torch.exp(difference/temperature)

    # final loss
    # sum_i log(1 + sum_{j!=i} exp(i*j+ - i*i+))
    losses_log = (1 + exp_dif.sum(-1)).log()

    losses = losses_log.mean(-1)

    check_for_nan_inf(losses)

    # ======================================================
    # stats
    # ======================================================
    anchor_negative = outterproduct*diagnol_zeros
    B = anchor_negative.shape[1]
    correct = (innerproduct >= anchor_negative).sum(-1).flatten(0,1) == B

    anchor_negative_mean = anchor_negative[anchor_negative > 0].mean().item()
    anchor_positive_mean=innerproduct.mean().item()
    positive_negative_difference = anchor_positive_mean - anchor_negative_mean
    stats=dict(
        anchor_negative=anchor_negative_mean,
        anchor_positive=anchor_positive_mean,
        positive_negative_difference=positive_negative_difference,
        accuracy=correct.float().mean().item(),
        )
    return losses, stats