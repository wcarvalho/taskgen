import math
import numpy as np

from collections import namedtuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer

TaskBufferInfo = namedtuple("TaskBufferInfo", ['start', 'end', 'length', 'batch_idx'])

class TrajectoryPrioritizedReplay(PrioritizedSequenceReplayBuffer):
    """docstring for TrajectoryPrioritizedReplay"""
    def __init__(self,
        max_episode_length=0,
        only_success_term=True,
        **kwargs):
        super(TrajectoryPrioritizedReplay, self).__init__(**kwargs)
        save__init__args(locals())
        # self.terminal_idxs = []
        # self.successfl_idxs = []

    # def append_samples(self, samples):
    #     """
    #     """
    #     T, idxs = super().append_samples(samples)


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




class TaskTracker(object):
    """docstring for TaskTracker"""
    def __init__(self):
        super(TaskTracker, self).__init__()
        self._data = []
        self.pntr = 0

    def append(self, **kwargs):
        self._data.append(TaskBufferInfo(**kwargs))

        # -----------------------
        # periodically cleanup stale data
        # -----------------------
        if self.pntr > 1e4:
            self._data = self._data[self.pntr:]
            self.pntr = 0

    def advance(self, stale_idx):
        """advance data pointer so initial datum is after current stale index.
        
        Args:
            stale_idx (TYPE): Description
        
        Raises:
            RuntimeError: Description
        """

        iteration = 0
        while self.pntr < len(self._data):
            iteration += 1
            start = self._data[self.pntr].start
            end = self._data[self.pntr].end
            between = start <= stale_idx  and stale_idx <= end
            if between:
                self.pntr += 1
            else:
                break

            # beyond viable data
            if self.pntr >= len(self._data):
                break

            if iteration > 100:
                raise RuntimeError("Infinite loop")


    def __repr__(self): return self.data.__repr__()

    @property
    def data(self):
        if self.pntr < len(self._data):
            return self._data[self.pntr:]
        else:
            return []



class MultiTaskReplayWrapper(object):
    """docstring for MultiTaskReplay"""
    def __init__(self, buffer, tasks, track_success_only=True):
        # super(MultiTaskReplay, self).__init__()
        self.buffer = buffer
        self.tasks = tasks
        self.track_success_only = track_success_only
        self.task_idxs = {t : TaskTracker() for idx, t in enumerate(self.tasks)}

        self.current_task = np.ones(self.B, dtype=np.int32)*-1
        self.current_start = np.ones(self.B, dtype=np.int32)*-1
        self.current_length = np.zeros(self.B, dtype=np.int32)
        self.num_traj = 0

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def append_samples(self, samples):
        T, idxs = self.buffer.append_samples(samples)

        if isinstance(idxs, np.ndarray):
            start = idxs[0]
            stop = idxs[-1] + 1
        else:
            start = idxs.start
            stop = idxs.stop
        # -----------------------
        # load tasks + which are done
        # -----------------------
        tasks = samples.samples.observation.mission_idx[0,:, 0].numpy()
        B = len(tasks)

        # done = samples.samples.done[-1]
        cumsum = np.cumsum(samples.samples.done, 0)
        final_min_idx = cumsum.argmin(0)
        # time-step a
        finished = start + final_min_idx + 1
        # either done is index after final non-done or index is last time-index
        done_idx = np.minimum(final_min_idx+1, T-1).numpy()
        done = samples.samples.done[done_idx, np.arange(B)]
        success = samples.samples.success[done_idx, np.arange(B)]

        # -----------------------
        # set indices which don't have task associated
        # -----------------------
        unset = self.current_task == -1
        self.current_task[unset] = tasks[unset]
        self.current_start[unset] = start

        self.current_length += done_idx + 1
        # -----------------------
        # when task finished, indx should get time
        # -----------------------
        # self.current_end[done] = finished[done].numpy()

        if done.sum():
            if self.track_success_only:
                iterator = list(filter(lambda b: done[b] and success[b], range(B)))
            else:
                iterator = list(filter(lambda b: done[b], range(B)))
            for b in iterator:
                t = tasks[b]
                end = self.current_start[b] + self.current_length[b]
                self.task_idxs[t].append(
                    start=self.current_start[b],
                    end=end,
                    batch_idx=b,
                    length=self.current_length[b],
                    )
            self.num_traj += len(iterator)
            self.current_task[done] = -1
            self.current_length[done] = 0

        if self._buffer_full:
            for info in self.task_idxs.values():
                info.advance(stop - 1) # actual idx is 1 before

        # from pprint import pprint
        # print(done)
        # print(self.current_task)
        # import ipdb; ipdb.set_trace()

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
        for t in self.tasks:
            options = self.task_idxs[t].data
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

        if len(anchor_infos) <= 1:
            # not enough data
            return None, {}

        batch_T = self.batch_T if batch_T is None else batch_T

        # ======================================================
        # conform time parameters to RNN storing interval
        # ======================================================
        max_T = (max_T // self.rnn_state_interval) * self.rnn_state_interval
        max_T = max(max_T, self.rnn_state_interval)
        max_length = (max_length // self.rnn_state_interval) * self.rnn_state_interval
        max_length = max(max_length, self.rnn_state_interval)

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
        augmented_B = 2*(augmented_B//2) # make sure divisible by 2

        if augmented_B < 2:
            raise RuntimeError("need bigger batches?")

        num_tasks = augmented_B//2
        # -----------------------
        # get time + batch indices
        # -----------------------
        anchor_idxes = np.random.choice(len(anchor_infos), size=num_tasks, replace=False)
        T_idxs = []
        B_idxs = []
        anchor_tasks = []
        
        def add(info):
            T_idxs.append(info.end - 1) # end should be a `done` state so start of next
            B_idxs.append(info.batch_idx)

        for idx in anchor_idxes:
            anchor_tasks.append(used_tasks[idx])
            add(anchor_infos[idx])
            add(positive_infos[idx])

        # batches will have ``augmented_T'' timesteps
        T_idxs = np.array(T_idxs) - augmented_T
        B_idxs = np.array(B_idxs)

        if self.rnn_state_interval > 1:  # Some rnn states stored; only sample those.
            T_idxs = (T_idxs // self.rnn_state_interval) * self.rnn_state_interval
        else:
            T_idxs = T_idxs

        # self.samples.done[T_idxs[0]:T_idxs[0] + augmented_T, B_idxs[0]]
        batch = self.extract_batch(T_idxs, B_idxs, augmented_T)

        sample_info =dict(
            tasks=np.array(anchor_tasks),
            batch_T=augmented_T,
        )

        return batch, sample_info


















class MultiTaskReplay(object):
    """docstring for MultiTaskReplay"""
    def __init__(self, buffer_kwargs, ReplayCls, size, tasks):
        super(MultiTaskReplay, self).__init__()
        save__init__args(locals())

        self.task2replay = {t: ReplayCls(size=size//len(tasks), **buffer_kwargs) for t in tasks}
        self.task2idx = {t:idx for idx, t in enumerate(self.tasks)}


    def append_samples(self, samples):
        tasks = samples.samples.observation.mission_idx[:,:, 0]
        T, B = tasks.shape[:2]

        # for t, idx in self.task2idx.items():

        # for b in range(tasks.shape[1]):
            # tasks[:, b]
            # import ipdb; ipdb.set_trace()

    def sample_trajectories(self, batch_B, batch_T=None, success_only=True):
        import ipdb; ipdb.set_trace()

    def sample_batch(self, batch_B, batch_T=None, success_only=False):
        import ipdb; ipdb.set_trace()








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
