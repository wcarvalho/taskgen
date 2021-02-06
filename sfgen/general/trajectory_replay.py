import numpy as np

from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer


class TrajectoryPrioritizedReplay(PrioritizedSequenceReplayBuffer):
    """docstring for TrajectoryPrioritizedReplay"""
    def __init__(self,
        only_success_term=True,
        **kwargs):
        super(TrajectoryPrioritizedReplay, self).__init__(**kwargs)
        save__init__args(locals())
        self.terminal_idxs = []
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

        options = self.samples.success.nonzero()
        num_options = len(options[0])
        if  num_options== 0:
            return None

        choices = np.random.randint(low=0, high=num_options, size=(batch_B,))

        total_size = batch_B*self.batch_T
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
