"""Utilities for keeping track of task trajectories in replay buffer

"""
from rlpyt.utils.quick_args import save__init__args

class TrajectoryInfo(object):
    """docstring for TrajectoryInfo"""
    def __init__(self, task, batch, start, absolute_start, length):
        super(TrajectoryInfo, self).__init__()
        save__init__args(locals())

    @property
    def end(self):
        return self.start + self.length

    @property
    def absolute_end(self):
        return self.absolute_start + self.length

    def __repr__(self): return dict(task=self.task, batch=self.batch, start=self.start, absolute_start=self.absolute_start, length=self.length).__repr__()

class TaskTracker(object):
    """docstring for TaskTracker"""
    def __init__(self, max_T):
        super(TaskTracker, self).__init__()
        self.last_idx = 0
        self.absolute_idx = 0
        self.max_T = max_T

    def update(self, task, batch, buffer_t):
        if buffer_t < self.last_idx:
            # have wrapped in buffer
            # actual index would be past max buffer size
            buffer_t = self.max_T + buffer_t

        length = buffer_t - self.last_idx + 1
        trajectory = TrajectoryInfo(
            task=task,
            batch=batch,
            start=self.last_idx,
            absolute_start=self.absolute_idx,
            length=length,
            )

        self.last_idx += length
        self.absolute_idx += length

        # when wrap over
        self.last_idx = self.last_idx % self.max_T

        return trajectory


class TrajectoryTracker(object):
    """docstring for TrajectoryTracker"""
    def __init__(self, max_T):
        super(TrajectoryTracker, self).__init__()
        self._data = []
        self.pntr = 0
        self.max_T = max_T

    def append(self, traj_info):
        self._data.append(traj_info)
        # -----------------------
        # periodically cleanup stale data
        # -----------------------
        if self.pntr > 1e4:
            self._data = self._data[self.pntr:]
            self.pntr = 0

    def advance(self, stale_idx):
        """If there's data, while the first data instance starts before the stale idx, keep advancing it
        
        Args:
            stale_idx (TYPE): what indices have been overwritten
        """

        iteration = 0
        while self.data and self.absolute_start <= stale_idx:
            iteration += 1
            self.pntr += 1

            if iteration > 100:
                raise RuntimeError("Infinite loop")


    def __repr__(self): return self.data.__repr__()

    @property
    def absolute_start(self):
        if self.data:
            return self._data[self.pntr].absolute_start
        else:
            return 1e10


    @property
    def data(self):
        if self.pntr < len(self._data):
            return self._data[self.pntr:]
        else:
            return []
