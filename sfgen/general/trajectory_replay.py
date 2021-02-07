import math
import numpy as np

from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer

def sample_TB(options, batch_B):
    num_options = len(options[0])
    choices = np.random.randint(low=0, high=num_options, size=(batch_B,))

    # import ipdb; ipdb.set_trace()
    T_idxs = np.array([options[0][c] for c in choices]) - batch_T
    B_idxs = np.array([options[1][c] for c in choices])


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

    def append_samples(self, samples):
        """
        """
        T, idxs = super().append_samples(samples)


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
