from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.replays.sequence.prioritized import UniformSequenceReplay

class TrajectoryUniformReplay(UniformSequenceReplayBuffer):
    """docstring for TrajectoryUniformReplay"""
    def __init__(self,
        only_success_term=True,
        **kwargs):
        super(TrajectoryUniformReplay, self).__init__(**kwargs)
        self.terminal_idxs = []
        # self.successfl_idxs = []

    def append_samples(self, samples):
        """
        """
        T, idxs = super().append_samples(samples)


        print("need to make sure you keep track of successful episodes somehow...")
        if self.only_success_term:
            import ipdb; ipdb.set_trace()
        else:
            raise NotImplementedError

class TrajectoryPrioritizedReplay(PrioritizedSequenceReplayBuffer):
    """docstring for TrajectoryPrioritizedReplay"""
    def __init__(self,
        only_success_term=True,
        **kwargs):
        super(TrajectoryPrioritizedReplay, self).__init__(**kwargs)
        self.terminal_idxs = []
        # self.successfl_idxs = []

    def append_samples(self, samples):
        """
        """
        T, idxs = super().append_samples(samples)


        print("need to make sure you keep track of successful episodes somehow...")
        if self.only_success_term:
            import ipdb; ipdb.set_trace()
        else:
            raise NotImplementedError