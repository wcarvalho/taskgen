import itertools
# from collections import namedtuple
# import copy
# import math
# import numpy as np
from rlpyt.agents.base import AgentInputs
from rlpyt.algos.pg.ppo import PPO
# from rlpyt.algos.pg.base import OptInfo as OptInfoRl
from rlpyt.utils.quick_args import save__init__args
# # from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
# from rlpyt.ul.replays.rl_with_ul_replay import (RlWithUlUniformReplayBuffer,
#     RlWithUlPrioritizedReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
# # from rlpyt.preloads.mlp import MlpModel
from rlpyt.utils.buffer import buffer_to, buffer_method
# from rlpyt.algos.utils import valid_from_done
# from rlpyt.preloads.utils import update_state_dict
# from rlpyt.ul.algos.utils.data_augs import random_shift
# from rlpyt.ul.preloads.rl.ul_models import UlEncoderModel
# from rlpyt.ul.preloads.ul.atc_models import ContrastModel
# from rlpyt.utils.logging import logger
# from rlpyt.ul.algos.utils.warmup_scheduler import GradualWarmupScheduler


# IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
# OptInfoUl = namedtuple("OptInfoUl", ["ulLoss", "ulAccuracy", "ulGradNorm",
    # "ulUpdates"])
# OptInfo = namedtuple("OptInfo", OptInfoRl._fields + OptInfoUl._fields)
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done", "prev_rnn_state", "success"])

from algos.trajectory_replay import TrajectoryUniformReplay
from utils.utils import consolidate_dict_list

class PPOAux(PPO):
    """docstring for PPOAux"""
    def __init__(self,
        AuxClasses=None,
        aux_kwargs=None,
        aux_epochs=20,
        # buffer args
        # prioritized_replay=True,
        warmup_T=0,
        batch_T=80,
        n_step_return=1,
        min_steps_learn=int(1e5),
        store_rnn_state_interval=1,
        replay_size=int(1e6),
        **kwargs):
        super(PPOAux, self).__init__(**kwargs)
        save__init__args(locals())

        self.aux_kwargs = aux_kwargs or dict()
        if self.initial_optim_state_dict:
            raise NotImplementedError("Need to load both ppo + aux task optimizer")
        assert store_rnn_state_interval > 0, "story rnn"

    def initialize(self, *args, examples=None, **kwargs):
        # ======================================================
        # old initialization
        # ======================================================
        super().initialize(*args, examples=examples, **kwargs)
        if self.linear_lr_schedule:
            raise NotImplementedError("Need to anneal learning rate for auxilliary task. Will need to compute total number of updates and all that jazz.")


        # ======================================================
        # Auxilliary tasks
        # ======================================================
        if self.AuxClasses is None: 
            self.aux_tasks = None
            return

        assert isinstance(self.AuxClasses, dict), 'please name each aux class'
        self.aux_tasks = {name: Cls(**self.aux_kwargs) for name, cls in self.AuxClasses.items()}
        self.aux_optimizers = {}
        for name, aux_task in self.aux_tasks:
            self.aux_optimizers[name] = self.OptimCls(
                itertools.chain(self.agent.parameters(), aux_task.parameters()),
                lr=self.learning_rate, **self.optim_kwargs)

        self.min_itr_learn_aux = int(self.min_steps_learn // self.batch_spec.size)
        # ======================================================
        # Replay Buffer
        # ======================================================
        if self.aux_tasks.use_replay_buffer:
            self.replay_buffer = self.initialize_replay_buffer(examples, self.batch_spec)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """Similar to DQN but uses replay buffers which return sequences, and
        stores the agent's recurrent state."""
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            success=examples["env_info"].success,
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
        # if self.prioritized_replay:
        # ReplayCls = TrajectoryPrioritizedReplay
        # else:
        #     ReplayCls = TrajectoryUniformReplay
        #     raise NotImplementedError

        return TrajectoryUniformReplay(**replay_kwargs)

    def optimize_agent(self, itr, samples):
        # ======================================================
        # Add samples to buffer
        # ======================================================
        if self.aux_tasks.use_replay_buffer:
            samples_to_buffer = SamplesToBuffer(
                observation=samples.env.observation,
                action=samples.agent.action,
                reward=samples.env.reward,
                done=samples.env.done,
                prev_rnn_state=samples.agent.agent_info.prev_rnn_state,
                success=samples.env.env_info.success,
            )
            self.replay_buffer.append_samples(samples_to_buffer)

        # ======================================================
        # PPO optimization
        # ======================================================
        opt_info = super().optimize_agent(itr, samples)

        info = dict(ppo=opt_info._asdict())
        if self.aux_tasks is None or itr < self.min_itr_learn_aux:
            return info

        # ======================================================
        # Auxilliary task optimization
        # ======================================================
        for aux_name, aux_task in self.aux_tasks.items():
            if aux_task.use_replay_buffer:
                aux_info = self.aux_replay_optimization(aux_name, aux_task, itr)
            else:
                aux_info = self.aux_samples_optimization(aux_name, aux_task, itr, samples)
            info[aux_name] = aux_info

        if self.linear_lr_schedule:
            raise NotImplementedError

        return info

    def aux_samples_optimization(self, aux_name, aux_task, itr, samples):
        raise NotImplementedError

    def aux_replay_optimization(self, aux_name, aux_task, itr):

        all_stats = []
        for epoch in range(self.aux_epochs):

            # -----------------------
            # sampling and preparing data
            # -----------------------
            if aux_task.use_trajectories:
                aux_samples = self.replay_buffer.sample_trajectories(batch_B=self.batch_spec.B)
                if aux_samples is None:
                    # nothing available
                    return info
            else:
                aux_samples = self.replay_buffer.sample_batch(batch_B=self.batch_spec.B)
            import ipdb; ipdb.set_trace()
            agent_inputs = AgentInputs(  # Move inputs to device once, index there.
                observation=aux_samples.all_observation,
                prev_action=aux_samples.all_action,
                prev_reward=aux_samples.all_reward,
            )
            agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
            if self.agent.recurrent:
                # Leave in [B,N,H] for slicing to minibatches.
                init_rnn_state = buffer_method(aux_samples.init_rnn_state, "transpose", 0, 1)
                init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            else:
                init_rnn_state = None


            # -----------------------
            # computations
            # -----------------------
            variables = self.agent.get_variables(*agent_inputs, init_rnn_state, all_variables=True)
            loss, stats = aux_task(variables)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.agent.parameters(), aux_task.parameters()), 
                self.clip_grad_norm)
            self.aux_optimizers[aux_name].step()

            all_stats.append(stats)

        return consolidate_dict_list(all_stats)
