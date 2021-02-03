import itertools
# from collections import namedtuple
# import copy
# import math
# import numpy as np

# from rlpyt.algos.pg.ppo import PPO
# from rlpyt.algos.pg.base import OptInfo as OptInfoRl
from rlpyt.utils.quick_args import save__init__args
# # from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
# from rlpyt.ul.replays.rl_with_ul_replay import (RlWithUlUniformReplayBuffer,
#     RlWithUlPrioritizedReplayBuffer)
# from rlpyt.utils.collections import namedarraytuple
# # from rlpyt.models.mlp import MlpModel
# from rlpyt.utils.buffer import buffer_to
# from rlpyt.algos.utils import valid_from_done
# from rlpyt.models.utils import update_state_dict
# from rlpyt.ul.algos.utils.data_augs import random_shift
# from rlpyt.ul.models.rl.ul_models import UlEncoderModel
# from rlpyt.ul.models.ul.atc_models import ContrastModel
# from rlpyt.utils.logging import logger
# from rlpyt.ul.algos.utils.warmup_scheduler import GradualWarmupScheduler


# IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
# OptInfoUl = namedtuple("OptInfoUl", ["ulLoss", "ulAccuracy", "ulGradNorm",
    # "ulUpdates"])
# OptInfo = namedtuple("OptInfo", OptInfoRl._fields + OptInfoUl._fields)
# SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    # ["observation", "action", "reward", "done"])



from sfgen.general.trajectory_replay import TrajectoryPrioritizedReplay, TrajectoryUniformReplay

class PPOAux(PPO):
    """docstring for PPOAux"""
    def __init__(self,
        AuxCls=None,
        aux_kwargs=None,
        # buffer args
        warmup_T=0,
        n_step_return=1,
        store_rnn_state_interval=10,
        replay_size=int(1e6),
        **kwargs):
        super(PPOAux, self).__init__(**kwargs)
        self.aux_kwargs = aux_kwargs or dict()
        save__init__args(locals())

    def initialize(self, *args, examples=None, **kwargs):
        # ======================================================
        # old initialization
        # ======================================================
        super().initialize(*args, examples=examples, **kwargs)
        if self.linear_lr_schedule:
            raise NotImplementedError("Need to anneal learning rate for auxilliary task. Will need to compute total number of updates and all that jazz.")


        # ======================================================
        # Auxilliary task
        # ======================================================
        if self.AuxCls is None: return

        self.aux_task = self.AuxCls(**self.aux_kwargs)
        # parameters = list(agent.parameters()) + 
        self.aux_optimizer = self.OptimCls(
            itertools.chain(agent.parameters(), self.aux_task.parameters())
            lr=self.learning_rate, **self.optim_kwargs)

        import ipdb; ipdb.set_trace()
        # ======================================================
        # Replay Buffer
        # ======================================================
        self.replay_buffer = self.initialize_replay_buffer(examples, batch_spec)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """Similar to DQN but uses replay buffers which return sequences, and
        stores the agent's recurrent state."""
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )

        if self.store_rnn_state_interval > 0:
            example_to_buffer = SamplesToBufferRnn(
                *example_to_buffer,
                prev_rnn_state=examples["agent_info"].prev_rnn_state,
            )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=self.store_rnn_state_interval,
            # batch_T fixed for prioritized, (relax if rnn_state_interval=1 or 0).
            batch_T=self.batch_T + self.warmup_T,
        )
        if self.prioritized_replay:
            ReplayCls = TrajectoryPrioritizedReplay
        else:
            ReplayCls = TrajectoryUniformReplay
            raise NotImplementedError

        return ReplayCls(**replay_kwargs)