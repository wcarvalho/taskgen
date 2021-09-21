
"""
Script for running individual experiments. Call from root.
Will call `build_and_train`.

To run:
    python launchers/starter/launch_individual.py

To run with breakpoint at exception:
    python -m ipdb -c continue launchers/starter/launch_individual.py

"""
# ======================================================
# Project wide code (change per project)
# ======================================================
import launchers.starter.individual_log as log
from launchers.starter.configs import configs, defaults



# ======================================================
# GENERIC CODE BELOW
# ======================================================
import multiprocessing
import os

import torch.cuda
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector)
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector)

# ======================================================
# RLPYT modules
# ======================================================
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger
from rlpyt.utils.logging.context import logger_context

# -----------------------
# loading model + agent
# -----------------------
from agents.babyai_agents import BabyAIR2d1Agent
# -----------------------
# auxilliary task modules
# -----------------------
from algos.r2d1_aux_joint import R2D1AuxJoint
from envs.rlpyt import babyai_utils

# ======================================================
# Our modules
# ======================================================
from envs.rlpyt.babyai_env import BabyAIEnv
from launchers.starter.configs import defaults
from nnmodules.babyai_model import BabyAIRLModel
from utils.runners import MinibatchRlEvalDict
from utils.runners import SuccessTrajInfo
# -----------------------
# loading configs
# -----------------------
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
    level,
    run_ID=0,
    cuda_idx=None,
    n_parallel=2,
    log_dir="logs",
    n_steps=5e5,
    log_interval_steps=2e5,
    snapshot_gap=10,
    **kwargs,
    ):
    
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
    log_dir = f"data/local/{log_dir}/{name}"

    # -----------------------
    # call train
    # -----------------------
    logger.set_snapshot_gap(snapshot_gap)
    train(config, affinity, log_dir, run_ID,
        name=name,
        gpu=gpu,
        parallel=parallel
        )

def train(config, affinity, log_dir, run_ID, name='babyai', gpu=False,
    parallel=True, skip_launched=True):

    # -----------------------
    # skip already run experiments
    # -----------------------
    subdir = os.path.join(log_dir, f"run_{run_ID}")
    if skip_launched and os.path.exists(subdir):
        print("="*25)
        print("Skipping:", subdir)
        print("="*25)
        return

    # ======================================================
    # load environment settings
    # ======================================================
    if config['settings']['env'] == "babyai":
        env_kwargs, eval_env_kwargs = babyai_utils.load_babyai_env(config)
    elif config['settings']['env'] == "babyai_kitchen":
        env_kwargs, eval_env_kwargs = babyai_utils.load_babyai_kitchen_env(config)
    else:
        raise RuntimeError("no env loaded")

    # ======================================================
    # load algorithm (loss functions) + agent (architecture)
    # ======================================================
    algo = R2D1AuxJoint(
        ReplayBufferCls=PrioritizedSequenceReplayBuffer,
        optim_kwargs=config['optim'],
        **config["algo"],
        )  # Run with defaults.

    agent = BabyAIR2d1Agent(
        **config['agent'],
        ModelCls=BabyAIRLModel,
        model_kwargs=config['model'],
        )

    # ======================================================
    # load sampler for collecting experience
    # ======================================================
    if gpu:
        sampler_class = GpuSampler
        CollectorCls = GpuResetCollector
    else:
        CollectorCls = CpuResetCollector
        if parallel: sampler_class = CpuSampler
        else:        sampler_class = SerialSampler

    sampler = sampler_class(
        EnvCls=BabyAIEnv,
        CollectorCls=CollectorCls,
        TrajInfoCls=SuccessTrajInfo,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        **config["sampler"]  # More parallel envs for batched forward-pass.
    )

    # ======================================================
    # Load runner + train
    # ======================================================
    runner = MinibatchRlEvalDict(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
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
    # ======================================================
    # env/agent settingns
    # ======================================================
    parser.add_argument('--level',
        help='BabyAI level',
        default='GoToRedBall')

    # ======================================================
    # run settings
    # ======================================================
    parser.add_argument('--cuda_idx',
        help='gpu to use ',
        type=int,
        default=None)
    parser.add_argument('--n_parallel',
        help='number of sampler workers',
        type=int,
        default=1)
    parser.add_argument('--n_steps',
        help='number of environment steps (default=1 million)',
        type=int,
        default=2e6)


    # ======================================================
    # logging
    # ======================================================
    parser.add_argument('--run_ID',
        help='run identifier (logging)',
        type=int,
        default=0)
    parser.add_argument('--log_dir',
        type=str,
        default='babyai')
    parser.add_argument('--log_interval_steps',
        help='Number of environment steps between logging to csv/tensorboard/etc (default=100 thousand)',
        type=int,
        default=1e5)
    parser.add_argument('--snapshot-gap',
        help='how often to save model',
        type=int,
        default=5)
    parser.add_argument('--verbosity',
        type=int,
        default=0)

    args = parser.parse_args()
    build_and_train(**vars(args))
