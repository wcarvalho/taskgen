import torch.nn
import torch.nn.functional as F


from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.algos.utils import valid_from_done
from rlpyt.models.mlp import MlpModel


from sfgen.tools.utils import consolidate_dict_list
from sfgen.tools.ops import check_for_nan_inf, duplicate_vector
from sfgen.general.loss_functions import mc_npair_loss


class AuxilliaryTask(torch.nn.Module):
    """docstring for AuxilliaryTask"""
    def __init__(self,
        epochs=5,
        batch_T=40,
        batch_B=0,
        sampler_bs=40,
        min_steps_learn=0,
        coeff=1e-3,
        **kwargs,
        ):
        super(AuxilliaryTask, self).__init__()
        save__init__args(locals())
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)

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

    @staticmethod
    def update_config(config):
        """Use this function to automatically set kwargs in config
        """
        pass

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
        
        **kwargs,
        ):
        super(ContrastiveHistoryComparison, self).__init__(**kwargs)
        save__init__args(locals())


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
        assert T == len(done), "done must cover same timespan"

        max_T = min(self.num_timesteps*self.dilation, T)

        num_timesteps = min(self.num_timesteps, T)

        segmented_history = history.view(T, 2, B//2, D)

        anchors = segmented_history[-max_T::self.dilation, 0]
        positives = segmented_history[-max_T::self.dilation, 1]

        if not variables.get('normalized_history', False):
            anchors = F.normalize(anchors + 1e-12, p=2, dim=-1)
            positives = F.normalize(positives + 1e-12, p=2, dim=-1)


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
            raise RuntimeError("Why did this happen?")
            # loss = losses.sum()*0
            # loss_scalar = 0
            # stats['no_valid'] = 1.0
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


class ContrastiveObjectModel(AuxilliaryTask):
    """Contrastive Object Model. Two options for negatives:
    1. random, just use random embeddings as negatives
    2. positives, use positives from *other* anchors as negatives for an anchor
    """
    def __init__(self,
        negatives='positives',
        temperature=0.01,
        history_dim=512,
        obs_dim=512,
        # nheads=8,
        max_actions=10,
        action_dim=64,
        nhidden=0,
        aux_nonlin='ReLU',
        **kwargs,
        ):
        super(ContrastiveObjectModel, self).__init__(**kwargs)
        save__init__args(locals())
        assert negatives in ['positives', 'random']

        # self.individual_dim = self.history_dim//self.nheads
        self.action_embedder = torch.nn.Embedding(
            num_embeddings=self.max_actions,
            embedding_dim=self.action_dim,
            )

        assert self.nhidden >= 0
        if self.nhidden == 0:
            self.model = torch.nn.Linear(self.action_dim + self.history_dim, self.obs_dim)
        else:
            self.model = MlpModel(
                input_size=self.action_dim + self.history_dim,
                hidden_sizes=[self.history_dim]*self.nhidden,
                output_size=self.obs_dim,
                nonlinearity=getattr(torch.nn, self.aux_nonlin),
            )

    def forward(self, variables, action, done, **kwargs):
        """Summary
        """
        object_histories = variables["goal_history"]
        object_observations = variables["goal"]
        T, B, N, D = object_histories.shape
        assert T == len(done), "done must cover same timespan"
        assert T == len(action), "done must cover same timespan"
        # ======================================================
        # Compute Model Predictions
        # ======================================================
        source_objects = object_histories[:-1]
        source_actions = action[:-1]
        # embed actions
        source_actions = self.action_embedder(source_actions)
        # T x B x D --> T x B x N x D
        source_actions = duplicate_vector(source_actions, n=N, dim=2)

        predictions = self.model(torch.cat((source_objects, source_actions), dim=-1))
        predictions = F.normalize(predictions + 1e-12, p=2, dim=-1)


        targets = object_observations[1:]


        # ======================================================
        # compute loss
        # ======================================================

        if self.negatives == "positives":
            losses, stats = mc_npair_loss(
                anchors=predictions.flatten(0,1),
                positives=targets.flatten(0,1),
                temperature=self.temperature)

        elif self.negatives == "random":
            raise NotImplementedError()


        valid = 1 - done[:-1].flatten(0,1).float()
        loss = valid_mean(losses, valid)
        loss_scalar = loss.item()
        # ======================================================
        # statistics
        # ======================================================
        stats.update(dict(
                    loss=loss_scalar,
                    loss_coeff=loss_scalar*self.coeff,
                    ))

        loss = self.coeff*loss
        return loss, stats

    @staticmethod
    def update_config(config):
        default_size = config['model']['default_size']
        nheads = config['model']['nheads']


        individual_rnn_dim = config['model'].get('individual_rnn_dim', None)
        if individual_rnn_dim is not None:
            history_dim = individual_rnn_dim
        else:
            history_size = default_size if default_size else config['model']['history_size']
            history_dim = history_size//nheads


        independent_compression = config['model']['independent_compression']
        obs_size = default_size if default_size else config['model']['goal_size']
        if independent_compression:
            obs_size = obs_size // nheads


        config['aux'].update(
            history_dim=history_dim,
            obs_dim=obs_size,
            nheads=nheads,
            )