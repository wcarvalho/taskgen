
import torch
import numpy as np

from rlpyt.utils.misc import zeros


def duplicate_vector(vector, n, dim=1):
  """B X D vector to B X N X D vector"""
  batch_size = vector.shape[:dim]
  return torch.ones(*batch_size, n, *vector.shape[dim:], device=vector.device)*vector.unsqueeze(dim)



def discount_cumulant_n_step(cumulant, done, n_step, discount, return_dest=None,
        done_n_dest=None, do_truncated=False, ncumulant_dims=0):
    """
    Major difference from RLPYT: can have a cumulant w/ more than zero dimensions.
    so reward might be [T,B] whereas cumulant might be [T,B,D1,...]

    Time-major inputs, optional other dimension: [T], [T,B], etc.  Computes
    n-step discounted returns within the timeframe of the of given cumulants. If
    `do_truncated==False`, then only compute at time-steps with full n-step
    future cumulants are provided (i.e. not at last n-steps--output shape will
    change!).  Returns n-step returns as well as n-step done signals, which is
    True if `done=True` at any future time before the n-step target bootstrap
    would apply (bootstrap in the algo, not here)."""
    rlen = cumulant.shape[0]
    if not do_truncated:
        rlen -= (n_step - 1)

    return_ = return_dest if return_dest is not None else zeros(
        (rlen,) + cumulant.shape[1:], dtype=cumulant.dtype).to(cumulant.device)
    done_n = done_n_dest if done_n_dest is not None else zeros(
        (rlen,) + cumulant.shape[1:-ncumulant_dims], dtype=done.dtype).to(cumulant.device)
    return_[:] = cumulant[:rlen]  # 1-step return is current cumulant.
    done_n[:] = done[:rlen]  # True at time t if done any time by t + n - 1
    is_torch = isinstance(done, torch.Tensor)
    if is_torch:
        done_dtype = done.dtype
        done_n = done_n.type(cumulant.dtype)
        done = done.type(cumulant.dtype)


    for _ in range(ncumulant_dims):
        done = done.unsqueeze(-1)
        done_n = done_n.unsqueeze(-1)

    if n_step > 1:
        if do_truncated:
            for n in range(1, n_step):
                return_[:-n] += (discount ** n) * cumulant[n:n + rlen] * (1 - done_n[:-n])
                done_n[:-n] = torch.max(done_n[:-n], done[n:n + rlen])
        else:
            for n in range(1, n_step):
                return_ += (discount ** n) * cumulant[n:n + rlen] * (1 - done_n)
                done_n[:] = torch.max(done_n, done[n:n + rlen])  # Supports tensors.

    return return_, done_n