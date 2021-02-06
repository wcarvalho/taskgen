import numpy as np
try:
    import wandb
except ModuleNotFoundError as e:
    print("Warning: `wandb is not available")
    pass
from rlpyt.utils.logging import logger
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.collections import namedarraytuple, AttrDict
from sfgen.tools.utils import flatten_dict

class MinibatchRlEvalDict(MinibatchRlEval):

    def initialize_logging(self):
        super().initialize_logging()
        self._opt_infos = None

    def store_diagnostics(self, itr, traj_infos, opt_info):
        """
        Store any diagnostic information from a training iteration that should
        be kept for the next logging iteration.
        """
        self._cum_completed_trajs += len(traj_infos)
        if not isinstance(opt_info, dict):
            opt_info = opt_info._asdict()
        new_data = flatten_dict(opt_info, sep="/")
        if self._opt_infos==None:
            self._opt_infos = new_data
        else:
            for k, v in self._opt_infos.items():
                new_v = new_data[k]
                v.extend(new_v if isinstance(new_v, list) else [new_v])

        # import ipdb; ipdb.set_trace()
        # for k, v in self._opt_infos.items():
        #     new_v = getattr(opt_info, k, [])
        #     v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)

    def _log_infos(self, traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k, [info[k] for info in traj_infos])

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = None # reset

class MinibatchRlEvalWandb(MinibatchRlEvalDict):
    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size
        self.wandb_info = {'cum_steps': cum_steps}
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
        wandb.log(self.wandb_info)

    def startup(self, *args, **kwargs):
        n_itr = super().startup(*args, **kwargs)
        wandb.watch(self.agent.model)
        return n_itr

    def _log_infos(self, traj_infos=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    values = [info[k] for info in traj_infos]
                    logger.record_tabular_misc_stat(k,
                                                    values)
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Median"] = np.median(values)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)





class SuccessTrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.success = False
        self.DiscountedReturn = 0
        self._cur_discount = 1

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.success = self.success or env_info.success
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount

    def terminate(self, observation):
        return self
