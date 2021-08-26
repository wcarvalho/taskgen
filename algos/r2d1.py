import torch

from rlpyt.agents.base import AgentInputs
from rlpyt.algos.dqn.r2d1 import SamplesToBufferRnn, PrioritiesSamplesToBuffer # algorithm
from rlpyt.algos.dqn.r2d1 import R2D1
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger
from rlpyt.algos.utils import valid_from_done

# from sfgen.general.r2d1_aux_joint import R2D1AuxJoint
from algos.trajectory_replay import TrajectoryPrioritizedReplay, TrajectoryUniformReplay, MultiTaskReplayWrapper

SamplesToBuffer_ = namedarraytuple("SamplesToBuffer_",
    SamplesToBufferRnn._fields + ("success",))

class R2D1v2(R2D1):
    """Prep for auxilliary tasks/gvfs
    R2D1 with support for (a) auxilliary tasks and (b) learning GVFs.
    GVFs must be defined in model.

    rewrite buffer initialization and getting samples for buffer
    so supported having "success" in buffer + samples
    also changed replay buffer to trajectory buffer
    """
    def __init__(self,
        AuxClasses=None,
        aux_kwargs=None,
        GvfCls=None,
        buffer_type="regular",
        gvf_kwargs=None,
        max_episode_length=0,
        train_tasks=None,
        joint=True,
        **kwargs):
        super(R2D1v2, self).__init__(**kwargs)

        save__init__args(locals())
        self.aux_kwargs = aux_kwargs or dict()
        self.gvf_kwargs = gvf_kwargs or dict()
        self.train_tasks = train_tasks or []

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """Similar to R2D1 except buffer storce episode success and have custom replay buffer which has capacity to sample trajectories."""
        example_to_buffer = SamplesToBuffer_(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            success=examples["env_info"].success,
            prev_rnn_state=examples["agent_info"].prev_rnn_state,
        )
        if self.store_rnn_state_interval == 0:
            raise NotImplementedError("Only handle storing rnn state")

        replay_kwargs = dict(
            example=example_to_buffer,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=self.store_rnn_state_interval,
            # batch_T fixed for prioritized, (relax if rnn_state_interval=1 or 0).
            batch_T=self.batch_T + self.warmup_T,
            max_episode_length=self.max_episode_length,
        )
        if self.prioritized_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
                input_priorities=self.input_priorities,  # True/False.
                input_priority_shift=self.input_priority_shift,
            ))
            ReplayCls = TrajectoryPrioritizedReplay
        else:
            ReplayCls = TrajectoryUniformReplay
        if self.ReplayBufferCls is not None:
            logger.log(f"WARNING: ignoring replay buffer class: {self.ReplayBufferCls} -- instead using {ReplayCls}")

        if self.buffer_type == 'regular':
            self.replay_buffer = ReplayCls(
                size=self.replay_size,
                **replay_kwargs
                )
        elif self.buffer_type == 'multitask':
            self.replay_buffer = MultiTaskReplayWrapper(
                buffer=ReplayCls(
                    size=self.replay_size,
                    **replay_kwargs
                    ),
                tasks=self.train_tasks,
                )
        else:
            raise NotImplementedError

        return self.replay_buffer

    def samples_to_buffer(self, samples):
        """Overwrote R2D1's samples_to_buffer class to include success. use THEIR parent (DQN) for original samples_to_buffer
        """
        samples_to_buffer = super(R2D1, self).samples_to_buffer(samples)
        if self.store_rnn_state_interval > 0:
            samples_to_buffer = SamplesToBuffer_(*samples_to_buffer,
                prev_rnn_state=samples.agent.agent_info.prev_rnn_state,
                success=samples.env.env_info.success,
                )
        else:
            raise NotImplementedError()
            # samples_to_buffer = SamplesToBuffer_(*samples_to_buffer,
            #     success=samples.env.env_info.success,
            #     )
        if self.input_priorities:
            priorities = self.compute_input_priorities(samples)
            samples_to_buffer = PrioritiesSamplesToBuffer(
                priorities=priorities, samples=samples_to_buffer)

        return samples_to_buffer

    def load_all_agent_inputs(self, samples, batch_T=0, warmup_T=-1):
        """Copied from R2D2:loss. Get inputs for model, target_model, + extra information such as actions taken, whether environment done was seen, etc.
        
        Args:
            samples (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        all_observation, all_action, all_reward = buffer_to(
            (samples.all_observation, samples.all_action, samples.all_reward),
            device=self.agent.device)


        batch_T = batch_T if batch_T else self.batch_T
        warmup_T = warmup_T if warmup_T >= 0 else self.warmup_T
        wT, bT, nsr = warmup_T, batch_T, self.n_step_return
        if wT > 0:
            warmup_slice = slice(None, wT)  # Same for agent and target.
            warmup_inputs = AgentInputs(
                observation=all_observation[warmup_slice],
                prev_action=all_action[warmup_slice],
                prev_reward=all_reward[warmup_slice],
            )
        agent_slice = slice(wT, wT + bT)
        agent_inputs = AgentInputs(
            observation=all_observation[agent_slice],
            prev_action=all_action[agent_slice],
            prev_reward=all_reward[agent_slice],
        )
        target_slice = slice(wT, None)  # Same start t as agent. (wT + bT + nsr)
        target_inputs = AgentInputs(
            observation=all_observation[target_slice],
            prev_action=all_action[target_slice],
            prev_reward=all_reward[target_slice],
        )
        action = samples.all_action[wT + 1:wT + 1 + bT]  # CPU.
        # return_ = samples.return_[wT:wT + bT]
        done = samples.done[wT:wT + bT]
        done_n = samples.done_n[wT:wT + bT]
        if self.store_rnn_state_interval == 0:
            init_rnn_state = None
        else:
            # [B,N,H]-->[N,B,H] cudnn.
            init_rnn_state = buffer_method(samples.init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
        if wT > 0:  # Do warmup.
            with torch.no_grad():
                _, target_rnn_state = self.agent.target(*warmup_inputs, init_rnn_state)
                _, init_rnn_state = self.agent(*warmup_inputs, init_rnn_state)
            # Recommend aligning sampling batch_T and store_rnn_interval with
            # warmup_T (and no mid_batch_reset), so that end of trajectory
            # during warmup leads to new trajectory beginning at start of
            # training segment of replay.
            warmup_invalid_mask = valid_from_done(samples.done[:wT])[-1] == 0  # [B]
            init_rnn_state[:, warmup_invalid_mask] = 0  # [N,B,H] (cudnn)
            target_rnn_state[:, warmup_invalid_mask] = 0
        else:
            target_rnn_state = init_rnn_state
        
        return agent_inputs, target_inputs, action, done, done_n, init_rnn_state, target_rnn_state