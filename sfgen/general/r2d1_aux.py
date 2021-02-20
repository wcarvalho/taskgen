import torch
import itertools

from rlpyt.agents.base import AgentInputs
from rlpyt.algos.dqn.r2d1 import R2D1, SamplesToBufferRnn, PrioritiesSamplesToBuffer # algorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.logging import logger
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done, discount_return_n_step


# from sfgen.general.r2d1_aux_joint import R2D1AuxJoint
from sfgen.general.trajectory_replay import TrajectoryPrioritizedReplay, TrajectoryUniformReplay, MultiTaskReplayWrapper
from sfgen.tools.ops import check_for_nan_inf
from sfgen.tools.utils import consolidate_dict_list, dictop
from sfgen.general.r2d1 import R2D1v2

SamplesToBuffer_ = namedarraytuple("SamplesToBuffer_",
    SamplesToBufferRnn._fields + ("success",))

class R2D1Aux(R2D1v2):

    def initialize(self, *args, examples=None, **kwargs):
        assert self.joint = False
        super().initialize(*args, examples=examples, **kwargs)

        # ======================================================
        # GVF
        # ======================================================
        self.gvf = None
        if self.GvfCls is not None:
            self.gvf = self.GvfCls(**self.gvf_kwargs)
            assert len(list(self.gvf.parameters())) == 0, 'all parameters should be in agent/model'
            self.gvf_optimizer = self.OptimCls(self.agent.parameters(),
                    lr=self.learning_rate, **self.optim_kwargs)

        # ======================================================
        # Auxilliary tasks
        # ======================================================
        if self.AuxClasses is None:
            self.aux_tasks = None
            return

        assert isinstance(self.AuxClasses, dict), 'please name each aux class and put in dictionary'
        aux_kwargs = dict(
            sampler_bs=self.sampler_bs,
            **self.aux_kwargs,
            )
        self.aux_tasks = {
            name:
                Cls(**aux_kwargs).to(self.agent.device)
                    for name, Cls in self.AuxClasses.items()
            }

        self.aux_optimizers = {}
        for name, aux_task in self.aux_tasks.items():
            if aux_task.use_trajectories:
                assert self.max_episode_length, "specify max episode length is sampling entire trajectories"
            if aux_task.has_parameters:
                params = itertools.chain(self.agent.parameters(), aux_task.parameters())
                if self.initial_optim_state_dict is not None:
                    raise RuntimeError("Need to setup loading parameters for aux task when has parameters")
            else:
                params = self.agent.parameters()
            self.aux_optimizers[name] = self.OptimCls(params,
                lr=self.learning_rate, **self.optim_kwargs)

    # ======================================================
    # Changed optimization so adds GVF + Aux tasks
    # ======================================================
    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        info = dict()
        # ======================================================
        # R2D1 optimization
        # ======================================================
        opt_info = super().optimize_agent(itr, samples, sampler_itr)
        info['r2d1'] = opt_info._asdict()

        if itr < self.min_itr_learn:
            return info

        # ======================================================
        # GVF Optimization
        # ======================================================
        if self.gvf:
            info['gvf'] = self.gvf_optimization(itr, sampler_itr)

        # ======================================================
        # Other Auxilliary Task Optimization
        # ======================================================
        if self.aux_tasks is None:
            return info

        for aux_name, aux_task in self.aux_tasks.items():

            if itr < aux_task.min_itr_learn:
                continue

            if aux_task.use_replay_buffer:
                aux_info = self.aux_replay_optimization(aux_name, aux_task, itr, sampler_itr)
            else:
                raise NotImplementedError("Auxilliary task that uses samples")
                # aux_info = self.aux_samples_optimization(aux_name, aux_task, itr, samples)
            info[aux_name] = aux_info

        return info

    def loss(self, samples):
        """Samples have leading Time and Batch dimentions [T,B,..]. Move all
        samples to device first, and then slice for sub-sequences.  Use same
        init_rnn_state for agent and target; start both at same t.  Warmup the
        RNN state first on the warmup subsequence, then train on the remaining
        subsequence.

        Returns loss (usually use MSE, not Huber), TD-error absolute values,
        and new sequence-wise priorities, based on weighted sum of max and mean
        TD-error over the sequence."""
        all_observation, all_action, all_reward = buffer_to(
            (samples.all_observation, samples.all_action, samples.all_reward),
            device=self.agent.device)
        wT, bT, nsr = self.warmup_T, self.batch_T, self.n_step_return
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
        return_ = samples.return_[wT:wT + bT]
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

        qs, _ = self.agent(*agent_inputs, init_rnn_state)  # [T,B,A]
        q = select_at_indexes(action, qs)
        with torch.no_grad():
            target_qs, _ = self.agent.target(*target_inputs, target_rnn_state)
            if self.double_dqn:
                next_qs, _ = self.agent(*target_inputs, init_rnn_state)
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1).values
            target_q = target_q[-bT:]  # Same length as q.

        disc = self.discount ** self.n_step_return
        y = self.value_scale(return_ + (1 - done_n.float()) * disc *
            self.inv_value_scale(target_q))  # [T,B]
        delta = y - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)


        # NOTE: by default, with R2D1, use squared-error loss, delta_clip=None.
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)


        if self.prioritized_replay:
            losses *= samples.is_weights.unsqueeze(0)  # weights: [B] --> [1,B]

        valid = valid_from_done(samples.done[wT:])  # 0 after first done.
        loss = valid_mean(losses, valid)
        td_abs_errors = abs_delta.detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)  # [T,B]
        valid_td_abs_errors = td_abs_errors * valid
        max_d = torch.max(valid_td_abs_errors, dim=0).values
        mean_d = valid_mean(td_abs_errors, valid, dim=0)  # Still high if less valid.
        priorities = self.pri_eta * max_d + (1 - self.pri_eta) * mean_d  # [B]

        check_for_nan_inf(q)
        return loss, td_abs_errors, priorities

    def gvf_optimization(self, itr, sampler_itr=None):
        """

        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr

        device = self.agent.device
        all_stats = []
        for _ in range(self.updates_per_optimize):
            samples = self.replay_buffer.sample_batch(self.batch_B, **self.gvf.batch_kwargs)
            if samples is None: # no available samples yet
                return {}
            self.gvf_optimizer.zero_grad()

            agent_inputs, _, action, done, done_n, init_rnn_state, _ = self.load_all_agent_inputs_prenstep(samples)

            variables = self.agent.get_variables(*agent_inputs, init_rnn_state, all_variables=True)
            target_variables = self.agent.get_variables(*target_inputs, init_target_rnn_state, target=True, all_variables=True)


            loss, stats = self.gvf(
                variables=variables,
                target_variables=target_variables,
                action=action.to(device), #starts with "prev action" so shift by 1
                done=done.to(device),
                batch_T=self.batch_T,
                n_step_return=self.n_step_return,
                discount=self.discount,
                )

            check_for_nan_inf(loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.clip_grad_norm)
            self.gvf_optimizer.step()

            all_stats.append(stats)

        return consolidate_dict_list(all_stats)

    def aux_replay_optimization(self, aux_name, aux_task, itr, sampler_itr=None):

        all_stats = []
        device=self.agent.device
        for epoch in range(aux_task.epochs):

            # ======================================================
            # sample batches
            # ======================================================
            batch_B = aux_task.batch_B if aux_task.batch_B else self.batch_B
            if aux_task.use_trajectories:
                samples, sample_info = self.replay_buffer.sample_trajectories(batch_B=batch_B, **aux_task.batch_kwargs)
                if samples is None:
                    # nothing available
                    return {}
                batch_T = sample_info['batch_T']
            else:
                samples = self.replay_buffer.sample_batch(batch_B=batch_B, **aux_task.batch_kwargs)
                sample_info = {}
                batch_T = self.batch_T

            self.aux_optimizers[aux_name].zero_grad()


            # ======================================================
            # prepare inputs
            # ======================================================
            agent_inputs, _, action, done, done_n, init_rnn_state, _ = self.load_all_agent_inputs_prenstep(samples, warmup_T=self.warmup_T, batch_T=batch_T)
            variables = self.agent.get_variables(*agent_inputs, init_rnn_state, all_variables=True)

            # ======================================================
            # loss + backprop
            # ======================================================
            loss, stats = aux_task(
                variables=variables,
                action=action.to(device),
                done=done.to(device),
                sample_info=sample_info,
                )
            check_for_nan_inf(loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.agent.parameters(), aux_task.parameters()), 
                self.clip_grad_norm)
            self.aux_optimizers[aux_name].step()

            all_stats.append(stats)

        return consolidate_dict_list(all_stats)

    def load_all_agent_inputs_prenstep(self, *args, **kwargs):
        return self.load_all_agent_inputs(*args, **kwargs, fewer_target_T=self.n_step_return)

    def load_all_agent_inputs(self, samples, batch_T=0, warmup_T=-1, extra_input_T=0, fewer_target_T=0):
        """Copies from R2D2:loss. Get inputs for model, target_model, + extra information such as actions taken, whether environment done was seen, etc.
        
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
        agent_slice = slice(wT, wT + bT + extra_input_T)
        agent_inputs = AgentInputs(
            observation=all_observation[agent_slice],
            prev_action=all_action[agent_slice],
            prev_reward=all_reward[agent_slice],
        )
        target_slice = slice(wT, wT + bT + nsr - fewer_target_T)  # Same start t as agent. (wT + bT + nsr)
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





def load_all_agent_inputs(agent, samples, batch_T=0, warmup_T=-1, n_step_return=5, store_rnn_state_interval=0, extra_input_T=0, fewer_target_T=0):
    """Copies from R2D2:loss. Get inputs for model, target_model, + extra information such as actions taken, whether environment done was seen, etc.
    
    Args:
        samples (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    all_observation, all_action, all_reward = buffer_to(
        (samples.all_observation, samples.all_action, samples.all_reward),
        device=agent.device)


    wT, bT, nsr = warmup_T, batch_T, n_step_return
    if wT > 0:
        warmup_slice = slice(None, wT)  # Same for agent and target.
        warmup_inputs = AgentInputs(
            observation=all_observation[warmup_slice],
            prev_action=all_action[warmup_slice],
            prev_reward=all_reward[warmup_slice],
        )
    agent_slice = slice(wT, wT + bT + extra_input_T)
    agent_inputs = AgentInputs(
        observation=all_observation[agent_slice],
        prev_action=all_action[agent_slice],
        prev_reward=all_reward[agent_slice],
    )
    target_slice = slice(wT, wT + bT + nsr - fewer_target_T)  # Same start t as agent. (wT + bT + nsr)
    target_inputs = AgentInputs(
        observation=all_observation[target_slice],
        prev_action=all_action[target_slice],
        prev_reward=all_reward[target_slice],
    )
    action = samples.all_action[wT + 1:wT + 1 + bT]  # CPU.
    # return_ = samples.return_[wT:wT + bT]
    done = samples.done[wT:wT + bT]
    done_n = samples.done_n[wT:wT + bT]
    if store_rnn_state_interval == 0:
        init_rnn_state = None
    else:
        # [B,N,H]-->[N,B,H] cudnn.
        init_rnn_state = buffer_method(samples.init_rnn_state, "transpose", 0, 1)
        init_rnn_state = buffer_method(init_rnn_state, "contiguous")
    if wT > 0:  # Do warmup.
        with torch.no_grad():
            _, target_rnn_state = agent.target(*warmup_inputs, init_rnn_state)
            _, init_rnn_state = agent(*warmup_inputs, init_rnn_state)
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