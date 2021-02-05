import torch.nn
from rlpyt.utils.quick_args import save__init__args


class GVF(torch.nn.Module):
    """docstring for GVF"""
    def __init__(self,
        ):
        super(GVF, self).__init__()
        save__init__args(locals())

    def forward(self, variables, target_variables, action, done_n):
        raise NotImplementedError


class GoalGVF(GVF):
    """docstring for GoalGVF"""

    def forward(self, variables, target_variables, action, done_n):
        import ipdb; ipdb.set_trace()
