import torch.nn
from rlpyt.utils.quick_args import save__init__args

class AuxilliaryTasks(torch.nn.Module):
    """docstring for AuxilliaryTasks"""
    def __init__(self, auxilliary_tasks):
        super(AuxilliaryTasks, self).__init__()
        self.auxilliary_tasks = auxilliary_tasks


class AuxilliaryTask(torch.nn.Module):
    """docstring for AuxilliaryTask"""
    def __init__(self,
        # use_replay_buffer=True,
        # use_trajectories=False,
        ):
        super(AuxilliaryTask, self).__init__()
        save__init__args(locals())

    @property
    def use_replay_buffer(self):
        return True

    @property
    def has_parameters(self):
        return False

    @property
    def use_trajectories(self):
        return False

    @property
    def batch_kwargs(self):
        return {}


class ContrastiveHistoryComparison(AuxilliaryTask):
    """docstring for ContrastiveHistoryComparison"""
    def __init__(self,
        success_only=True,
        ):
        super(ContrastiveHistoryComparison, self).__init__()
        save__init__args(locals())

    def forward(self, variables):
        import ipdb; ipdb.set_trace()
        return 0, {}

    @property
    def use_trajectories(self):
        return True

    @property
    def batch_kwargs(self):
        return dict(
            success_only=self.success_only,
            )
    

# class GoalGVF(AuxilliaryTask):
#     """docstring for GoalGVF"""
#     def __init__(self, arg):
#         super(GoalGVF, self).__init__()
#         self.arg = arg

#     def forward(self, variables):
#         import ipdb; ipdb.set_trace()
#         return 0, {}

#     @property
#     def use_trajectories(self):
#         return False
