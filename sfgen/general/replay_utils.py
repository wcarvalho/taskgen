"""Utilities for keeping track of task trajectories in replay buffer

"""
from rlpyt.utils.quick_args import save__init__args

class TrajectoryInfo(object):
    """docstring for TrajectoryInfo"""
    def __init__(self, task, batch, absolute_start, length, max_T):
        super(TrajectoryInfo, self).__init__()
        save__init__args(locals())

    @property
    def start(self):
        return self.absolute_start % self.max_T

    @property
    def final_idx(self):
        return (self.start + self.length - 1) % self.max_T

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
        self.max_T = max_T

    def update(self, task, batch, absolute_t):
        """track trajectories using absolute indices
        
        Args:
            task (TYPE): Description
            batch (TYPE): Description
            absolute_t (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        length = absolute_t - self.last_idx + 1
        trajectory = TrajectoryInfo(
            task=task,
            batch=batch,
            absolute_start=self.last_idx,
            length=length,
            max_T=self.max_T,
            )

        self.last_idx += length

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


    def __repr__(self): 
        if self.data:
            return self.data[0].__repr__()
        else:
            return str(None)

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
