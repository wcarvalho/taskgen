import torch.nn
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.algos.utils import valid_from_done

class AuxilliaryTasks(torch.nn.Module):
    """docstring for AuxilliaryTasks"""
    def __init__(self, auxilliary_tasks):
        super(AuxilliaryTasks, self).__init__()
        self.auxilliary_tasks = auxilliary_tasks


class AuxilliaryTask(torch.nn.Module):
    """docstring for AuxilliaryTask"""
    def __init__(self,
        epochs=5,
        batch_T=40,
        batch_B=0,
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

    loss_per_timestep = losses_log.mean(-1)

    return loss_per_timestep
class ContrastiveHistoryComparison(AuxilliaryTask):
    """docstring for ContrastiveHistoryComparison"""
    def __init__(self,
        success_only=True,
        max_T=150,
        temperature=0.01,
        num_timesteps=1,
        **kwargs,
        ):
        super(ContrastiveHistoryComparison, self).__init__(**kwargs)
        save__init__args(locals())
        self.iter = 0


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

        segmented_history = history.view(T, 2, B//2, D)

        anchors = segmented_history[:-self.num_timesteps, 0]
        positives = segmented_history[:-self.num_timesteps, 1]

        if not variables['normalized_history']:
            anchors = F.normalize(anchors, p=2, dim=-1)
            positives = F.normalize(positives, p=2, dim=-1)


        loss_per_timestep = (mc_npair_loss(anchors, positives, self.temperature) + mc_npair_loss(positives, anchors, self.temperature))/2

        reversed_done = torch.flip(done[:-self.num_timesteps], (0,))
        # cum sum to when get > 0 done in reverse (1st done going correct way)
        done_gt1 = (reversed_done.cumsum(0) > 0).float()
        # reverse again and have mask
        valid = torch.flip(1 - done_gt1, (0,))
        valid = valid.prod(-1) # multiply batchwise. if any is invalid at time-step, all are

        loss = valid_mean(loss_per_timestep, valid)

        # ======================================================
        # statistics
        # ======================================================

        correct = (innerproduct >= anchor_negative).sum(-1).flatten(0,1) == (B//2)
        stats=dict(
            loss=loss.item(),
            anchor_negative=anchor_negative[anchor_negative > 0].mean().item(),
            anchor_positive=innerproduct.mean().item(),
            accuracy=correct.float().mean().item(),
            num_tasks=B//2,
            length=(1-done.float()).sum(0).mean().item(),
            )

        self.iter += 1

        if self.iter > 50:
            import ipdb; ipdb.set_trace()

        return loss, stats

    @property
    def use_trajectories(self):
        return True

    @property
    def batch_kwargs(self):
        return dict(
            success_only=self.success_only,
            max_T=self.max_T,
            )
