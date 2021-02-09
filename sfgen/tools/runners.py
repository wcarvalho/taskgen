import numpy as np
try:
    import wandb
except ModuleNotFoundError as e:
    print("Warning: `wandb is not available")
    pass
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.collections import namedarraytuple, AttrDict
from sfgen.tools.utils import flatten_dict

class MinibatchRlEvalDict(MinibatchRlEval):

    def __init__(self, eval_tasks=None, **kwargs):
        super().__init__(**kwargs)
        save__init__args(locals())
        self.eval_tasks = eval_tasks or []

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

        self.pbar.update((itr + 1) % self.log_interval_itrs)

    def _log_infos(self, traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        def log_info(infos, keys, start=''):
            for k in keys:
                key = f"{start}/{k}" if start else k
                data = [info[k] for info in infos]
                record_tabular_misc_stat(key, data)

        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            keys = list(filter(lambda k: not k.startswith("_"), traj_infos[0].keys()))
            if self.eval_tasks:
                train_info = list(filter(lambda t: not t['_task'][0] in self.eval_tasks, traj_infos))
                log_info(train_info, keys, start='train')

                eval_info = list(filter(lambda t:t['_task'][0] in self.eval_tasks, traj_infos))

                if eval_info:
                    log_info(eval_info, keys, start='eval')
                    for idx, eval_task in enumerate(self.eval_tasks):
                        eval_info = list(filter(lambda t:t['_task'][0] in [eval_task], traj_infos))
                        log_info(eval_info, keys, start=f'eval_{idx}')
            else:
                log_info(traj_infos, keys)


        if self._opt_infos:
            for k, v in self._opt_infos.items():
                record_tabular_misc_stat(k, v)
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
                    record_tabular_misc_stat(k,
                                                    values)
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Median"] = np.median(values)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)



# ======================================================
# tracking and logging data
# ======================================================
def record_tabular_misc_stat(key, values, placement='back'):
    if placement == 'front':
        prefix = ""
        suffix = key
    else:
        prefix = key
        suffix = ""
        if logger._tf_summary_writer is not None:
            prefix += "/"  # Group stats together in Tensorboard.
    if len(values) > 0:
        logger.record_tabular(prefix + "Average" + suffix, np.average(values))
        # logger.record_tabular(prefix + "Std" + suffix, np.std(values))
        # logger.record_tabular(prefix + "Median" + suffix, np.median(values))
        logger.record_tabular(prefix + "Min" + suffix, np.min(values))
        logger.record_tabular(prefix + "Max" + suffix, np.max(values))
    else:
        logger.record_tabular(prefix + "Average" + suffix, np.nan)
        # logger.record_tabular(prefix + "Std" + suffix, np.nan)
        # logger.record_tabular(prefix + "Median" + suffix, np.nan)
        logger.record_tabular(prefix + "Min" + suffix, np.nan)
        logger.record_tabular(prefix + "Max" + suffix, np.nan)


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
        # self.Return = 0
        # self.NonzeroRewards = 0
        self.success = False
        self.DiscountedReturn = 0
        self._cur_discount = 1
        self._task = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        # self.Return += reward
        # self.NonzeroRewards += reward != 0
        self.success = self.success or env_info.success
        self.DiscountedReturn += self._cur_discount * reward
        self._task = observation.mission_idx
        self._cur_discount *= self._discount

    def terminate(self, observation):
        return self
