
"""
Script for running individual experiments. Call from root.
Will call `build_and_train`.

To run:
    python launchers/sfgen_babyai/launch_individual.py

To run with breakpoint at exception:
    python -m ipdb -c continue launchers/sfgen_babyai/launch_individual.py

"""
import copy
import multiprocessing
import os
import json
import torch.cuda
import yaml

# ======================================================
# RLPYT modules
# ======================================================
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler


from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector)
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector)

from rlpyt.utils.logging.context import logger_context
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer

from rlpyt.utils.logging import logger
# ======================================================
# BabyAI/Minigrid modules
# ======================================================
import babyai.utils

# ======================================================
# Our modules
# ======================================================
from envs.rlpyt.babyai_env import BabyAIEnv
from envs.rlpyt import babyai_utils

from utils.runners import SuccessTrajInfo
from utils.runners import MinibatchRlEvalDict

# -----------------------
# loading model + agent
# -----------------------
from agents.babyai_agents import BabyAIR2d1Agent
from nnmodules.babyai_model import BabyAIRLModel
from nnmodules.sfgen_model import SFGenModel

# -----------------------
# auxilliary task modules
# -----------------------
from nnmodules.history_aux import ContrastiveHistoryComparison, ContrastiveObjectModel
from nnmodules.gvfs import GoalGVF
from algos.r2d1_aux import R2D1Aux
from algos.r2d1_aux_joint import R2D1AuxJoint

# -----------------------
# loading configs
# -----------------------
from utils.variant import update_config
import launchers.sfgen_babyai.individual_log as log
from launchers.sfgen_babyai.configs import configs, defaults

def load_config(settings):

    # which config setting to use
    env = settings.get("env", defaults['env_configs'])
    model = settings.get("model", defaults['model_configs'])
    algorithm = settings.get("algorithm", defaults['algorithm_configs'])
    aux = settings.get("aux", defaults['aux_configs'])
    gvf = settings.get("gvf", defaults['gvf_configs'])

    # load configs
    config = configs['env_configs'][env]
    config = update_config(config, configs['model_configs'][model])
    config = update_config(config, configs['algorithm_configs'][algorithm])
    config = update_config(config, configs['aux_configs'][aux])
    config = update_config(config, configs['gvf_configs'][gvf])

    return config

def load_algo_agent(config, algo_kwargs=None, agent_kwargs=None, horizon=100, train_tasks=None):
    """Summary
    
    Args:
        config (TYPE): Description
        algo_kwargs (None, optional): Description
        agent_kwargs (None, optional): Description
    
    Raises:
        NotImplementedError: Description
    """
    algo_kwargs = algo_kwargs or {}
    agent_kwargs = agent_kwargs or {}
    train_tasks = train_tasks or []


    # -----------------------
    # model
    # -----------------------
    if config['settings']['model'] in ['babyai', 'chaplot']:
        ModelCls = BabyAIRLModel
    elif config['settings']['model'] in ['sfgen']:
        ModelCls = SFGenModel
    else: raise NotImplementedError



    # -----------------------
    # gvf
    # -----------------------
    GvfCls = None
    if config['settings']['gvf'] == 'none': pass
    elif config['settings']['gvf'] in ['goal_gvf']:
        GvfCls = GoalGVF
    else: raise NotImplementedError


    # ======================================================
    # R2D1
    # ======================================================
    rlhead = config['model']['rlhead']
    if not rlhead in ['dqn']:
        print("="*40)
        print("Algorithm:", config['settings']['algorithm'])
        print(f"Warning: changing head {rlhead} to 'dqn'")
        print("="*40)
        config['model']['rlhead'] = 'dqn'


    algo_kwargs['max_episode_length'] = horizon
    algo_kwargs['GvfCls'] = GvfCls
    algo_kwargs['gvf_kwargs'] = config['gvf']
    algo_kwargs['aux_kwargs'] = config['aux']
    algo_kwargs['train_tasks'] = train_tasks


    algo = R2D1AuxJoint(
        ReplayBufferCls=PrioritizedSequenceReplayBuffer,
        optim_kwargs=config['optim'],
        **config["algo"],
        **algo_kwargs,
        )  # Run with defaults.

    agent = BabyAIR2d1Agent(
        **config['agent'],
        ModelCls=ModelCls,
        model_kwargs=config['model'],
        **agent_kwargs,
        )

    # buffer_type = config['algo'].get("buffer_type", 'regular')

    return algo, agent

def build_and_train(
    level,
    run_ID=0,
    cuda_idx=None,
    n_parallel=2,
    log_dir="logs",
    n_steps=5e5,
    log_interval_steps=2e5,
    num_missions=0,
    snapshot_gap=10,
    verbosity=0,
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

    config['env'].update(
        dict(
            num_missions=num_missions,
            verbosity=verbosity,
            ))

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
    env_kwargs, eval_env_kwargs = babyai_utils.load_babyai_kitchen_env(config['env'])

    eval_levels = list(eval_env_kwargs['level_kwargs'].keys())

    # ======================================================
    # load algorithm (loss functions) + agent (architecture)
    # ======================================================
    algo, agent = load_algo_agent(config, horizon=horizon,
        train_tasks=train_tasks)

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
        eval_keys=eval_levels,
        eval_identifier='level',
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
    parser.add_argument('--agent',
        help='which config to load',
        type=str,
        default='ppo')
    parser.add_argument('--level',
        help='BabyAI level',
        default='GoToRedBall')
    parser.add_argument('--num_missions',
        help='number of missions to sample (default 0 = infinity)',
        type=int,
        default=0)

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
