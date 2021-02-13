import torch.nn
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.algos.utils import valid_from_done
from sfgen.tools.utils import consolidate_dict_list
from sfgen.tools.ops import check_for_nan_inf
from sfgen.general.loss_functions import mc_npair_loss


class AuxilliaryTask(torch.nn.Module):
    """docstring for AuxilliaryTask"""
    def __init__(self,
        epochs=5,
        batch_T=40,
        batch_B=0,
        sampler_bs=40,
        # use_replay_buffer=True,
        # use_trajectories=False,
        **kwargs,
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
        max_T=150,
        num_timesteps=1,
        temperature=0.01,
        min_trajectory=50,
        dilation=1,
        symmetric=True,
        min_steps_learn=0,
        **kwargs,
        ):
        super(ContrastiveHistoryComparison, self).__init__(**kwargs)
        save__init__args(locals())
        self.min_itr_learn = int(self.min_steps_learn // self.sampler_bs)


    def forward(self, variables, action, done, sample_info, **kwargs):
        """Summary
        i+ = positive for data at index i

        Args:
            variables (TYPE): Description
            action (TYPE): Description
            done (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        history = variables['goal_history']
        T, B, D = history.shape
        assert B%2 == 0, "should be divisible by 2"
        assert T == len(done), "all must cover same timespan"

        max_T = min(self.num_timesteps*self.dilation, T)

        num_timesteps = min(self.num_timesteps, T)

        segmented_history = history.view(T, 2, B//2, D)

        anchors = segmented_history[-max_T::self.dilation, 0]
        positives = segmented_history[-max_T::self.dilation, 1]

        if not variables.get('normalized_history', False):
            anchors = F.normalize(anchors + 1e-6, p=2, dim=-1)
            positives = F.normalize(positives + 1e-6, p=2, dim=-1)


        losses_1, stats_1 = mc_npair_loss(anchors, positives, self.temperature)
        if self.symmetric:
            losses_2, stats_2 = mc_npair_loss(positives, anchors, self.temperature)
            losses = (losses_1 + losses_2)/2
            stats = consolidate_dict_list([stats_1, stats_2])
        else:
            losses = losses_1
            stats = stats_1

        reversed_done = torch.flip(done.float(), (0,))
        # cum sum to when get > 0 done in reverse (1st done going correct way)
        done_gt1 = (reversed_done.cumsum(0) > 1).float()
        # reverse again and have mask
        # multiply batchwise. if any is invalid at t, all batches at t are
        valid = torch.flip(1 - done_gt1, (0,)).prod(-1) 
        valid = valid[-max_T::self.dilation]

        if valid.sum() == 0:
            loss = losses.sum()*0
            loss_scalar = 0
            stats['no_valid'] = 1.0
        else:
            loss = valid_mean(losses, valid)
            loss_scalar = loss.item()


        # ======================================================
        # statistics
        # ======================================================
        stats.update(dict(
                    loss=loss_scalar,
                    num_tasks=B//2,
                    length=valid.sum(0).mean().item(),
                    ))

        return loss, stats

    @property
    def use_trajectories(self):
        return True

    @property
    def batch_kwargs(self):
        return dict(
            success_only=self.success_only,
            max_T=self.max_T,
            min_trajectory=self.min_trajectory,
            )
