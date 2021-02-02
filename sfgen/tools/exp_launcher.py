import os
import os.path as osp
import time
import datetime
from rlpyt.utils.launching.affinity import get_n_run_slots
from rlpyt.utils.logging.context import get_log_dir
from rlpyt.utils.launching.exp_launcher import log_exps_tree, log_num_launched, launch_experiment

def get_run_name(log_dir):
    return log_dir.split("/")[4]
    # return "/".join(log_dir.split("/")[4:6])

def get_log_dir(experiment_name, root_log_dir=None, date=True):
    yyyymmdd_hhmmss = datetime.datetime.today().strftime("%Y.%m.%d-%H.%M.%S")
    yyyymmdd, hhmmss = yyyymmdd_hhmmss.split("-")
    root_log_dir = LOG_DIR if root_log_dir is None else root_log_dir
    log_dir = osp.join(root_log_dir, "local", yyyymmdd, hhmmss, experiment_name)
    return log_dir


def run_experiments(script, affinity_code, experiment_title, runs_per_setting,
        variants, log_dirs, common_args=None, runs_args=None,
        set_egl_device=False, root_log_dir=None):
    """Call in a script to run a set of experiments locally on a machine.  Uses
    the ``launch_experiment()`` function for each individual run, which is a 
    call to the ``script`` file.  The number of experiments to run at the same
    time is determined from the ``affinity_code``, which expresses the hardware
    resources of the machine and how much resource each run gets (e.g. 4 GPU
    machine, 2 GPUs per run).  Experiments are queued and run in sequence, with
    the intention to avoid hardware overlap.  Inputs ``variants`` and ``log_dirs``
    should be lists of the same length, containing each experiment configuration
    and where to save its log files (which have the same name, so can't exist
    in the same folder).

    Hint:
        To monitor progress, view the `num_launched.txt` file and `experiments_tree.txt`
        file in the experiment root directory, and also check the length of each
        `progress.csv` file, e.g. ``wc -l experiment-directory/.../run_*/progress.csv``.
    """
    n_run_slots = get_n_run_slots(affinity_code)
    exp_dir = get_log_dir(experiment_title, root_log_dir=root_log_dir)
    procs = [None] * n_run_slots
    common_args = () if common_args is None else common_args
    assert len(variants) == len(log_dirs)
    if runs_args is None:
        runs_args = [()] * len(variants)
    assert len(runs_args) == len(variants)
    log_exps_tree(exp_dir, log_dirs, runs_per_setting)
    num_launched, total = 0, runs_per_setting * len(variants)
    for run_ID in range(runs_per_setting):
        for variant, log_dir, run_args in zip(variants, log_dirs, runs_args):
            launched = False
            log_dir = osp.join(exp_dir, log_dir)
            os.makedirs(log_dir, exist_ok=True)
            while not launched:
                for run_slot, p in enumerate(procs):
                    if p is None or p.poll() is not None:
                        procs[run_slot] = launch_experiment(
                            script=script,
                            run_slot=run_slot,
                            affinity_code=affinity_code,
                            log_dir=log_dir,
                            variant=variant,
                            run_ID=run_ID,
                            args=common_args + run_args,
                            set_egl_device=set_egl_device,
                        )
                        launched = True
                        num_launched += 1
                        log_num_launched(exp_dir, num_launched, total)
                        break
                if not launched:
                    time.sleep(10)
    for p in procs:
        if p is not None:
            p.wait()  # Don't return until they are all done.
