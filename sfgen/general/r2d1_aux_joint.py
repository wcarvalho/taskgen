import torch
import collections
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


from sfgen.general.trajectory_replay import TrajectoryPrioritizedReplay, TrajectoryUniformReplay, MultiTaskReplayWrapper
from sfgen.tools.ops import check_for_nan_inf
from sfgen.tools.utils import consolidate_dict_list, dictop

SamplesToBuffer_ = namedarraytuple("SamplesToBuffer_",
    SamplesToBufferRnn._fields + ("success",))

class R2D1AuxJoint(R2D1):
    """R2D1 with support for (a) auxilliary tasks and (b) learning GVFs.
    GVFs must be defined in model.
    """

    # ======================================================
    # changed initialization so adds GVF + Aux tasks
    # ======================================================
    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        # ======================================================
        # original initialization
        # ======================================================
        self.agent = agent
        self.n_itr = n_itr
        self.sampler_bs = sampler_bs = batch_spec.size
        self.mid_batch_reset = mid_batch_reset
        self.updates_per_optimize = max(1, round(self.replay_ratio * sampler_bs /
            self.batch_size))
        logger.log(f"From sampler batch size {batch_spec.size}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        eps_itr_max = max(1, int(self.eps_steps // sampler_bs))
        agent.set_epsilon_itr_min_max(self.min_itr_learn, eps_itr_max)
        self.initialize_replay_buffer(examples, batch_spec)

        # ======================================================
        # Load GVF
        # ======================================================
        self.gvf = None
        if self.GvfCls is not None:
            self.gvf = self.GvfCls(**self.gvf_kwargs)
            assert len(list(self.gvf.parameters())) == 0, 'all parameters should be in agent/model'
        # ======================================================
        # Load Auxilliary tasks
        # ======================================================
        self.aux_tasks = None
        if self.AuxClasses is None:
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

        # ======================================================
        # Optimizer
        # ======================================================
        self.optimizer = self.OptimCls(self.all_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)

        import ipdb; ipdb.set_trace()

    def all_parameters(self):
        params = [self.agent.parameters()]
        for name, aux_task in self.aux_tasks.items():
            params.append(aux_task.parameters())
        return itertools.chain(*params)

    # ======================================================
    # Changed optimization so adds GVF + Aux tasks
    # ======================================================
    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Copied and editted from R2D1
        """
        # ======================================================
        # add to replay
        # ======================================================
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)

        if itr < self.min_itr_learn:
            return {}

        # ======================================================
        # optimization
        # ======================================================
        all_stats = []

        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_B)
            self.optimizer.zero_grad()
            total_loss = 0

            # -----------------------
            # get data
            # -----------------------
            agent_inputs, target_inputs, action, done, done_n, init_rnn_state, target_rnn_state = load_all_agent_inputs(self.agent, samples_from_replay,
                batch_T=self.batch_T, warmup_T=self.warmup_T, n_step_return=self.n_step_return, store_rnn_state_interval=self.store_rnn_state_interval)

            # qs, _ = self.agent(*agent_inputs, init_rnn_state)  # [T,B,A]
            variables = self.agent.get_variables(*agent_inputs, init_rnn_state, all_variables=True)
            qs = variables['q']

            # target_qs, _ = self.agent.target(*target_inputs, target_rnn_state)
            target_variables = self.agent.get_variables(*target_inputs, init_target_rnn_state, target=True, all_variables=True)
            target_qs = target_variables['q']

            # next_qs, _ = self.agent(*target_inputs, init_rnn_state)
            next_variables = self.agent.get_variables(*target_inputs, init_rnn_state, all_variables=True)
            next_qs = variables['q']

            # ======================================================
            # losses
            # ======================================================
            info = dict()
            # -----------------------
            # r2d1 loss
            # -----------------------
            r2d1_loss, td_abs_errors, priorities, info['r2d1'] = self.r2d1_loss(qs, target_qs, next_qs, action, done, done_n)
            total_loss = total_loss + r2d1_loss


            # -----------------------
            # gvf
            # -----------------------
            if self.gvf:
                gvf_loss, info['gvf'] = self.gvf(
                    variables=variables,
                    target_variables=target_variables,
                    action=action.to(device), #starts with "prev action" so shift by 1
                    done=done.to(device),
                    batch_T=self.batch_T,
                    n_step_return=self.n_step_return,
                    discount=self.discount,
                    )
                total_loss = total_loss + gvf_loss

            # -----------------------
            # aux tasks
            # -----------------------
            for aux_name, aux_task in self.aux_tasks.items():
                if itr < aux_task.min_itr_learn:
                    continue
                assert aux_task.use_replay_buffer
                assert aux_task.use_trajectories == False
                aux_loss, info[aux_name] = aux_task(
                    variables=variables,
                    action=action.to(device),
                    done=done.to(device),
                    sample_info=sample_info,
                    )
                total_loss = total_loss + aux_loss


            import ipdb; ipdb.set_trace()

            # -----------------------
            # optimization set
            # -----------------------
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.all_parameters(), self.clip_grad_norm)
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(priorities)
            info['r2d1']['gradNorm'].append(grad_norm.item())

            all_stats.append(info)

            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target()
        self.update_itr_hyperparams(itr)

        all_stats = consolidate_dict_list(all_stats)


        import ipdb; ipdb.set_trace()
        return



    def r2d1_loss(self, qs, target_qs, next_qs, action, done, done_n):
        """Samples have leading Time and Batch dimentions [T,B,..]. Move all
        samples to device first, and then slice for sub-sequences.  Use same
        init_rnn_state for agent and target; start both at same t.  Warmup the
        RNN state first on the warmup subsequence, then train on the remaining
        subsequence.

        Returns loss (usually use MSE, not Huber), TD-error absolute values,
        and new sequence-wise priorities, based on weighted sum of max and mean
        TD-error over the sequence."""
        stats = collections.defaultdict(list)

        # qs, _ = self.agent(*agent_inputs, init_rnn_state)  # [T,B,A]
        q = select_at_indexes(action, qs)
        with torch.no_grad():
            # target_qs, _ = self.agent.target(*target_inputs, target_rnn_state)
            if self.double_dqn:
                # next_qs, _ = self.agent(*target_inputs, init_rnn_state)
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

        # -----------------------
        # info on loss
        # -----------------------
        check_for_nan_inf(loss)
        stats['loss'].append(r2d1_loss.item())
        stats['tdAbsErr'].extend(list(td_abs_errors[::8].numpy()))
        stats['priority'].extend(list(priorities))

        return loss, td_abs_errors, priorities, stats


