import torch.nn
from rlpyt.utils.quick_args import save__init__args

from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done, discount_return_n_step

from sfgen.tools.utils import dictop
from sfgen.tools.ops import discount_cumulant_n_step

class GVF(torch.nn.Module):
    """docstring for GVF"""
    def __init__(self,
        ):
        super(GVF, self).__init__()
        save__init__args(locals())

    def forward(self, variables, target_variables, **kwargs):
        raise NotImplementedError

    @property
    def batch_kwargs(self):
        return {}


class GoalGVF(GVF):
    """docstring for GoalGVF"""
    def __init__(self,
        cumulant='goal',
        success_only=True,
        batch_T=None,
        n_step_return=None,
        discount=None,
        ):
        super(GVF, self).__init__()
        save__init__args(locals())

    def forward(self, variables, target_variables, action, done, batch_T, n_step_return=None, discount=None, **kwargs):
        stats = dict()

        # -----------------------
        # load settings
        # -----------------------
        n_step_return = n_step_return if n_step_return else self.n_step_return
        # batch_T = batch_T if batch_T else self.batch_T
        discount = discount if discount else self.discount
        assert n_step_return is not None, "either algo sets n_step_return or loaded in settings"
        # assert batch_T is not None, "either algo sets batch_T or loaded in settings"
        assert discount is not None, "either algo sets discount or loaded in settings"

        # ======================================================
        # compute return
        # ======================================================
        cumulant = variables[self.cumulant]
        assert len(cumulant) == batch_T, "data should only cover batch_T"

        # length bT - nstep
        return_n, done_n = discount_cumulant_n_step(
            cumulant=cumulant, # use first batch_T inputs to compute returns
            ncumulant_dims=1,
            done=done,
            n_step=n_step_return,
            discount=discount,
            do_truncated=False,
        )
        num_preds = batch_T - n_step_return
        return_ = return_n[:num_preds]
        done_n_ = done_n[:num_preds]


        # ======================================================
        # compute y: return + gamma^n target
        # ======================================================
        target_goal_predictions = target_variables['goal_predictions'][n_step_return:]
        target_actions = action[n_step_return:n_step_return+num_preds]
        target_predictions = select_at_indexes(target_actions, target_goal_predictions)

        disc = discount ** n_step_return
        # return(t:t+n) + gamma^n*pred(n)
        y = return_ + (1 - done_n_) * disc* target_predictions

        # ======================================================
        # get predictions
        # ======================================================
        goal_predictions = variables['goal_predictions'][:num_preds]
        predictions = select_at_indexes(action[:num_preds], goal_predictions)

        # ======================================================
        # loss
        # ======================================================
        delta = y - predictions
        losses = 0.5 * delta ** 2
        valid = valid_from_done(done)  # 0 after first done.
        loss = valid_mean(losses, valid)


        # ======================================================
        # store some stats
        # ======================================================
        stats.update(dict(
            delta=delta.mean().item(),
            cumulant=cumulant.mean().item(),
            goal_predictions=predictions.mean().item(),
            target_goal_predictions=target_predictions.mean().item(),
            loss=loss.item(),
            ))

        return loss, stats


    @property
    def batch_kwargs(self):
        return dict(
            success_only=self.success_only,
            )
    