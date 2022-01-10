
"""
Script for running individual experiments. Call from root.
Will call `build_and_train`.

To run:
    python projects/starter/launch_individual.py

To run with breakpoint at exception:
    python -m ipdb -c continue projects/starter/launch_individual.py

"""
# ==================================================
# Project wide code (change per project)
# ==================================================
import projects.thor_nav.individual_log as log
from projects.thor_nav.configs import configs, defaults
from projects.thor_nav.configs import defaults
from projects.thor_nav.log_fn import thor_nav_log_fn

# ==================================================
# loading env, agent, model
# ==================================================# 
from rlpyt.algos.pg.ppo import PPO
from agents.babyai_agents import BabyAIPPOAgent
from nnmodules.thor_resnet_model import ThorModel

from envs.rlpyt.thor_env import ThorEnv, ThorTrajInfo
from envs.thor_nav import ALFRED_CONSTANTS

# ==================================================
# GENERIC CODE BELOW
# ==================================================
import multiprocessing
import os
import collections

import torch.cuda
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector)
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector)

from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger
from rlpyt.utils.logging.context import logger_context

from utils.runners import MinibatchRlEvalDict
from utils.variant import update_config



def load_config(settings):
  config = dict()
  for key, default_config in defaults.items():
    # get from settings or from defaults
    default_config_key = defaults[key]
    default_config = configs[key][default_config_key]
    update = settings.get(key, default_config)

    # update
    config = update_config(config, update)

  return config


def build_and_train(
    run_ID=0,
    cuda_idx=None,
    n_parallel=1,
    log_dir="single",
    snapshot_gap=10,
    skip_launched=False,
    **kwargs,
    ):
    """setup experiment details and run train."""
    # -----------------------
    # use individua_log.config to load settings
    # -----------------------
    settings = log.config.get("settings", {})
    # load default settings
    config = load_config(settings)
    # override settings using log file
    config = update_config(config, log.config)


    # -----------------------
    # whether to use gpu
    # -----------------------
    gpu=cuda_idx is not None and torch.cuda.is_available()
    print("="*20)
    print(f"Using GPU: {gpu}")
    print("="*20)

    # -----------------------
    # setting parallel processing
    # -----------------------
    n_parallel = min(n_parallel, multiprocessing.cpu_count())
    affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    parallel = len(affinity['workers_cpus']) > 1

    # -----------------------
    # setting diectory
    # -----------------------
    settings = config['settings']
    name = f"{settings['algorithm']}__{settings['model']}__{settings['env']}"
    log_dir = f"data/thor_nav/local/{log_dir}/{name}"

    # -----------------------
    # call train
    # -----------------------
    logger.set_snapshot_gap(snapshot_gap)
    train(config, affinity, log_dir, run_ID,
        name=name,
        gpu=gpu,
        parallel=parallel,
        skip_launched=False,
        )

def train(config, affinity, log_dir, run_ID, name='thor', gpu=False,
    parallel=True, skip_launched=True):
    """Shared by launch_individual and launch_batch
    
    Args:
        config (TYPE): Description
        affinity (TYPE): Description
        log_dir (TYPE): Description
        run_ID (TYPE): Description
        name (str, optional): Description
        gpu (bool, optional): Description
        parallel (bool, optional): Description
        skip_launched (bool, optional): Description
    
    Returns:
        TYPE: Description
    """
    # -----------------------
    # skip already run experiments
    # -----------------------
    subdir = os.path.join(log_dir, f"run_{run_ID}")
    if skip_launched and os.path.exists(subdir):
        print("="*25)
        print("Skipping:", subdir)
        print("="*25)
        return

    # ==================================================
    # load environment settings
    # ==================================================
    env_kwargs = config['env']
    env_kwargs['floorplans']= ALFRED_CONSTANTS.TRAIN_SCENE_NUMBERS

    eval_env_kwargs = config.get('eval_env', env_kwargs)
    eval_env_kwargs['floorplans']= ALFRED_CONSTANTS.TEST_SCENE_NUMBERS


    # ==================================================
    # load algorithm (loss functions) + agent (architecture)
    # ==================================================
    algo = PPO(
        optim_kwargs=config['optim'],
        **config["algo"],
        )  # Run with defaults.

    agent = BabyAIPPOAgent(
        **config['agent'],
        ModelCls=ThorModel,
        model_kwargs=config['model'],
        )

    # ==================================================
    # load sampler for collecting experience
    # ==================================================
    if gpu:
        sampler_class = GpuSampler
        CollectorCls = GpuResetCollector
    else:
        CollectorCls = CpuResetCollector
        if parallel: sampler_class = CpuSampler
        else:        sampler_class = SerialSampler

    sampler = sampler_class(
        EnvCls=ThorEnv,
        CollectorCls=CollectorCls,
        TrajInfoCls=ThorTrajInfo,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        **config["sampler"]  # More parallel envs for batched forward-pass.
    )

    # ==================================================
    # Load runner + train
    # ==================================================
    runner = MinibatchRlEvalDict(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        log_fn=thor_nav_log_fn,
        **config["runner"],
    )

    with logger_context(
        log_dir,
        run_ID,
        name,
        config,
        snapshot_mode="last+gap",
        override_prefix=True,
        use_summary_writer=True,
        ):
        runner.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ==================================================
    # run settings
    # ==================================================
    parser.add_argument('--cuda_idx',
        help='gpu to use ',
        type=int,
        default=None)
    parser.add_argument('--n_parallel',
        help='number of sampler workers',
        type=int,
        default=1)


    # ==================================================
    # logging
    # ==================================================
    parser.add_argument('--run_ID',
        help='run identifier (logging)',
        type=int,
        default=0)

    parser.add_argument('--snapshot-gap',
        help='how often to save model',
        type=int,
        default=5)
    parser.add_argument('--verbosity',
        type=int,
        default=0)
    parser.add_argument('--skip-launched',
        type=int,
        default=0)

    args = parser.parse_args()
    build_and_train(**vars(args))
