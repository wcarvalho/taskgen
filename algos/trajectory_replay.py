import torch
import math
import numpy as np


from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer


from algos.replay_utils import TaskTracker, TrajectoryTracker


class TrajectoryPrioritizedReplay(PrioritizedSequenceReplayBuffer):
    """simple wrapper around PrioritizedSequenceReplayBuffer.
    If want to only sample successful episodes, have own sampling function. Otherwise, reverts to PrioritizedSequenceReplayBuffer.
    """
    def __init__(self,
        max_episode_length=0,
        only_success_term=True,
        **kwargs):
        super(TrajectoryPrioritizedReplay, self).__init__(**kwargs)
        save__init__args(locals())


    def sample_trajectories(self, batch_B, batch_T=None, success_only=True):
        """Can dynamically input length of sequences to return, by ``batch_T``,
        else if ``None`` will use interanlly set value.  Returns batch with
        leading dimensions ``[batch_T, batch_B]``.
        """
        batch_T = self.batch_T if batch_T is None else batch_T

        if success_only:
            options = self.samples.success.nonzero()
            import ipdb; ipdb.set_trace()
        else:
            options = self.samples.done.nonzero()
            import ipdb; ipdb.set_trace()
        num_options = len(options[0])
        if  num_options== 0:
            return None

        import ipdb; ipdb.set_trace()


        choices = np.random.randint(low=0, high=num_options, size=(batch_B,))

        # import ipdb; ipdb.set_trace()
        T_idxs = np.array([options[0][c] for c in choices]) - batch_T
        B_idxs = np.array([options[1][c] for c in choices])

        if self.rnn_state_interval > 0:  # Some rnn states stored; only sample those.
            T_idxs = (T_idxs // self.rnn_state_interval) * self.rnn_state_interval
        return self.extract_batch(T_idxs, B_idxs, batch_T)

    def sample_batch(self, batch_B, batch_T=None, success_only=False):
        if not success_only: return super().sample_batch(batch_B)
        batch_T = self.batch_T if batch_T is None else batch_T


        # ======================================================
        # first sample successful terminal time-steps
        # ======================================================
        options = self.samples.success.nonzero()
        num_options = len(options[0])
        if  num_options== 0:
            return None
        choices = np.random.randint(low=0, high=num_options, size=(batch_B,))

        # ======================================================
        # an episode covers k segments of length batch_T
        # pick which segment to sample by picking how far back to go
        # from done
        # ======================================================
        num_segments_in_episode = math.ceil(self.max_episode_length/batch_T)
        prior_segment_steps = np.random.randint(low=0, high=num_segments_in_episode, size=(batch_B,))

        # ======================================================
        # compute time/batch indices
        # ======================================================
        T_idxs = np.array([options[0][c] for c in choices])
        B_idxs = np.array([options[1][c] for c in choices])

        #                           at least 1 to go 1 segment before ending of episode
        T_idxs_s = T_idxs - batch_T*(1+prior_segment_steps)

        if self.rnn_state_interval > 1:  # Some rnn states stored; only sample those.
            T_idxs_r = (T_idxs_s // self.rnn_state_interval) * self.rnn_state_interval
        else:
            T_idxs_r = T_idxs_s


        # ======================================================
        # don't want to go into ``impossible'' data
        # ======================================================
        if not self._buffer_full:
            T_idxs_r = np.maximum(T_idxs_r, 0)

        batch = self.extract_batch(T_idxs_r, B_idxs, batch_T)

        return batch







class MultiTaskReplayWrapper(object):
    """docstring for MultiTaskReplay"""
    def __init__(self, buffer, tasks, track_success_only=True):
        # super(MultiTaskReplay, self).__init__()
        self.buffer = buffer
        self.tasks = tasks
        self.track_success_only = track_success_only
        self.task_2_traj = {t : TrajectoryTracker(max_T=self.T) for idx, t in enumerate(self.tasks)}
        self.batch_2_task = {b : TaskTracker(max_T=self.T) for b in range(self.B)}

        self.num_traj = 0
        self.absolute_idx = 0

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def append_samples(self, samples):
        T, idxs = self.buffer.append_samples(samples)

        if isinstance(idxs, np.ndarray):
            start = idxs[0]
            # stop = idxs[-1] + 1
        else:
            start = idxs.start
            # stop = idxs.stop

        # -----------------------
        # buffer data
        # -----------------------
        buffer_tasks = self.samples.observation.mission_idx
        buffer_done = self.samples.done

        # -----------------------
        # batch data
        # -----------------------
        tasks = samples.samples.observation.mission_idx[:,:,0].numpy()
        dones = samples.samples.done.numpy()
        success = samples.samples.success

        # ======================================================
        # Add data
        # ======================================================
        done = dones.nonzero()
        for t,b in zip(done[0], done[1]):

            traj_info = self.batch_2_task[b].update(
                task=tasks[t,b],
                batch=b,
                absolute_t=self.absolute_idx+t,
                )
            if self.track_success_only:
                if success[t,b]:
                    self.task_2_traj[traj_info.task].append(traj_info)
                    self.num_traj += 1
            else:
                self.task_2_traj[traj_info.task].append(traj_info)
                self.num_traj += 1

            task = traj_info.task
            batch = traj_info.batch

            assert buffer_tasks[traj_info.start, batch, 0] == task
            assert buffer_tasks[traj_info.final_idx, batch, 0] == task
            assert buffer_done[traj_info.final_idx, batch]


        # -----------------------
        # update absolute indx
        # -----------------------
        self.absolute_idx += T

        # ======================================================
        # advance buffers past stale data
        # ======================================================
        if self._buffer_full:
            # every idx before (current position - length of buffer) is invalid (i.e. overwritten)
            stale_idx = self.absolute_idx - self.T
            for task, info in self.task_2_traj.items():
                info.advance(stale_idx)
                assert info.absolute_start > stale_idx
                if info.data:
                    assert buffer_tasks[info.data[0].start, info.data[0].batch, 0] == task






    def sample_trajectories(self, batch_B, batch_T=None, max_T=150, success_only=True, min_trajectory=50):

        if self.num_traj < min_trajectory:
            return None, {}

        # ======================================================
        # collect pairs of tasks
        # ======================================================
        anchor_infos = []
        positive_infos = []
        used_tasks = []
        max_length = 0
        min_length = 1e10
        for t in self.tasks:
            options = self.task_2_traj[t].data
            num_options = len(options)
            if num_options < 2:
                continue
            choices = np.random.choice(num_options, size=(2,), replace=False) # get anchor, positive
            anchor_info = options[choices[0]]
            positive_info = options[choices[1]]

            anchor_infos.append(anchor_info)
            positive_infos.append(positive_info)
            used_tasks.append(t)

            max_length = max(max_length, anchor_info.length, positive_info.length)
            min_length = min(min_length, anchor_info.length, positive_info.length)

        if len(anchor_infos) <= 1:
            logger.log(f"skipped trajectory sampling: not enough pairs: {len(anchor_infos)}")
            return None, {}

        batch_T = self.batch_T if batch_T is None else batch_T

        # ======================================================
        # conform time parameters to RNN storing interval
        # ======================================================
        max_T = (max_T // self.rnn_state_interval) * self.rnn_state_interval
        max_T = max(max_T, self.rnn_state_interval)
        max_length = (max_length // self.rnn_state_interval) * self.rnn_state_interval
        max_length = max(max_length, self.rnn_state_interval)
        # min_length = (min_length // self.rnn_state_interval) * self.rnn_state_interval


        # ======================================================
        # get indices
        # ======================================================
        # -----------------------
        # get number of tasks to use
        # -----------------------
        max_size = batch_T*batch_B
        tasks_B = 2*len(anchor_infos) # at most 1 (anchor, positive) per task w 2 available
        augmented_T = min(max_length, max_T)
        length_B = (max_size//(2*augmented_T)) # might need to subsample if beyond cap
        augmented_B = min(length_B, tasks_B)

        if augmented_B < 2:
            raise RuntimeError("need bigger batches?")

        num_tasks = augmented_B//2
        # -----------------------
        # get time + batch indices
        # -----------------------
        anchor_idxes = np.random.choice(len(anchor_infos), size=num_tasks, replace=False)
        T_idxs = []
        B_idxs = []
        tasks = []
        
        def add(info):
            # end should be index ONE after `done` state
            T_idxs.append(info.end)
            B_idxs.append(info.batch)

        for idx in anchor_idxes:
            tasks.extend([used_tasks[idx]]*2)
            add(anchor_infos[idx])
            add(positive_infos[idx])

        # batches will have ``augmented_T'' timesteps
        T_idxs = np.array(T_idxs)
        B_idxs = np.array(B_idxs)
        tasks = torch.from_numpy(np.array(tasks))

        T_idxs_a = T_idxs - augmented_T
        if self.rnn_state_interval > 1:  # Some rnn states stored; only sample those.
            raise RuntimeError("book-keeping is annoying. haven't solved this case")
            T_idx_r = (T_idxs_a // self.rnn_state_interval) * self.rnn_state_interval
        else:
            T_idx_r = T_idxs_a

        T_idx_r = T_idx_r % self.T

        batch = self.extract_batch(T_idx_r, B_idxs, augmented_T)

        task_mask = batch.all_observation.mission_idx[:augmented_T,:,0] == tasks.unsqueeze(0)
        sample_info =dict(
            tasks=tasks,
            task_mask=task_mask,
            batch_T=augmented_T,
        )


        if task_mask[-1].sum() < len(tasks):
            # self.samples.observation.mission_idx[T_idxs, B_idxs, 0]
            # self.samples.observation.mission_idx[anchor_infos[4].start, anchor_infos[4].batch, 0]
            raise NotImplementedError("shouldn't happen. probably bug in tracking tasks.")

        return batch, sample_info



















class TrajectoryUniformReplay(UniformSequenceReplayBuffer):
    """docstring for TrajectoryUniformReplay"""
    def __init__(self,
        # only_success_term=True,
        **kwargs):
        super(TrajectoryUniformReplay, self).__init__(**kwargs)
        save__init__args(locals())
        raise NotImplemented("Never finished implementing")
        # self.terminal_idxs = []
        # self.successfl_idxs = []

    def sample_trajectories(self, batch_B, batch_T=None):
        """Can dynamically input length of sequences to return, by ``batch_T``,
        else if ``None`` will use interanlly set value.  Returns batch with
        leading dimensions ``[batch_T, batch_B]``.
        """
        batch_T = self.batch_T if batch_T is None else batch_T

        options = self.samples.success.nonzero()
        num_options = len(options[0])
        if  num_options== 0:
            return None

        choices = np.random.randint(low=0, high=num_options, size=(batch_B,))

        # import ipdb; ipdb.set_trace()
        T_idxs = np.array([options[0][c] for c in choices]) - batch_T
        B_idxs = np.array([options[1][c] for c in choices])

        if self.rnn_state_interval > 0:  # Some rnn states stored; only sample those.
            T_idxs = (T_idxs // self.rnn_state_interval) * self.rnn_state_interval
        return self.extract_batch(T_idxs, B_idxs, batch_T)

    # def append_samples(self, samples):
    #     """
    #     """
    #     T, idxs = super().append_samples(samples)


        # if self.only_success_term:
        #     if samples.success.sum() == 0: return
        #     print("need to make sure you keep track of successful episodes somehow...")
        #     import ipdb; ipdb.set_trace()
        # else:
        #     raise NotImplementedError
