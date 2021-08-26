
"""
Script for running individual experiments.
Call from root

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
from utils import SuccessTrajInfo

# -----------------------
# loading model + agent
# -----------------------
from agents.babyai_agents import BabyAIR2d1Agent
from sfgen.babyai.babyai_model import BabyAIRLModel
from pytorch.sfgen_model import SFGenModel

# -----------------------
# auxilliary task modules
# -----------------------
from pytorch.history_aux import ContrastiveHistoryComparison, ContrastiveObjectModel
from pytorch.gvfs import GoalGVF
from algos.r2d1_aux import R2D1Aux
from algos.r2d1_aux_joint import R2D1AuxJoint

# -----------------------
# loading configs
# -----------------------
from sfgen.babyai.configs import algorithm_configs, model_configs, env_configs, aux_configs, gvf_configs
from utils import update_config
import launchers.sfgen.individual_log as log

def load_config(settings,
    default_env='babyai_kitchen',
    default_model='sfgen',
    default_algorithm='r2d1',
    default_aux='none',
    default_gvf='none',
    ):
    env = settings.get("env", default_env)
    model = settings.get("model", default_model)
    algorithm = settings.get("algorithm", default_algorithm)
    aux = settings.get("aux", default_aux)
    gvf = settings.get("gvf", default_gvf)

    config = env_configs[env]
    config = update_config(config, model_configs[model])
    config = update_config(config, algorithm_configs[algorithm])
    config = update_config(config, aux_configs[aux])
    config = update_config(config, gvf_configs[gvf])

    return config

def build_and_train(
    level="pong",
    run_ID=0,
    cuda_idx=None,
    n_parallel=2,
    log_dir="logs",
    n_steps=5e5,
    log_interval_steps=2e5,
    num_missions=0,
    snapshot_gap=10,
    model='sfgen',
    algorithm='r2d1',
    env='babyai_kitchen',
    verbosity=0,
    **kwargs,
    ):
    
    # use log.config to try to load settings
    settings = log.config.get("settings", {})
    config = load_config(settings, env, model, algorithm)
    config = update_config(config, log.config)

    config['env'].update(
        dict(
            num_missions=num_missions,
            verbosity=verbosity,
            ))


    gpu=cuda_idx is not None and torch.cuda.is_available()
    print("="*20)
    print(f"Using GPU: {gpu}")
    print("="*20)

    n_parallel = min(n_parallel, multiprocessing.cpu_count())
    affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))

    settings = config['settings']
    name = f"{settings['algorithm']}__{settings['model']}__{settings['env']}"
    log_dir = f"data/local/{log_dir}/{name}"

    parallel = len(affinity['workers_cpus']) > 1

    logger.set_snapshot_gap(snapshot_gap)
    train(config, affinity, log_dir, run_ID,
        name=name,
        gpu=gpu,
        parallel=parallel
        )

def load_instr_preprocessor(path="preloads/babyai/vocab.json"):
    instr_preprocessor = babyai.utils.format.InstructionsPreprocessor(path=path)

    path = instr_preprocessor.vocab.path
    if not os.path.exists(path):
        raise RuntimeError(f"Please create vocab and put in {path}")
    else:
        print(f"Successfully loaded {path}")

    return instr_preprocessor

def load_task_indices(path="preloads/babyai/tasks.json"):
    if not os.path.exists(path):
        print(f"No task index information found at: {path}")
        return {}

    with open(path, 'r') as f:
        try:
            task2idx = json.load(f)
        except Exception as e:
            print("="*25)
            print(f"Couldn't load: {path}")
            print(e)
            print("="*25)
            return {}

    print(f"Successfully loaded {path}")
    return task2idx

def load_task_info(config, tasks_path='experiments/task_setups', rootdir='.', kitchen_kwargs=dict(tile_size=0)):
    task_file = config['env'].get('task_file', None)
    if task_file:
        with open(os.path.join(rootdir, tasks_path, task_file), 'r') as f:
          tasks = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        raise NotImplementedError("implement (1) loading possible train tasks and setting corresponding variables. probably just match yaml file format? `load_kitchen_tasks` will need to support having empty `objects` field. will need to generalize that later ")

    if isinstance(tasks, dict):
        env = tasks.get('env', config['settings']['env'])
        if env == 'babyai_kitchen':
            from envs.babyai_kitchen.tasks import load_kitchen_tasks

        train, train_kinds = load_kitchen_tasks(tasks.get('train'), kitchen_kwargs=kitchen_kwargs)
        test, eval_kinds = load_kitchen_tasks(tasks.get('test', None), kitchen_kwargs=kitchen_kwargs)
        config['env']['valid_tasks'] = train
        config['eval_env']['valid_tasks'] = list(set(train+test))

        config['level']['task_kinds'] = list(set(train_kinds+eval_kinds))

    else:
        raise RuntimeError(f"Don't know how to load: {tasks}")


    return train, test

def load_aux_tasks(config):

    aux_tasks = config['settings']['aux']
    if isinstance(aux_tasks, str):
        if aux_tasks == 'none': return None
        aux_tasks = [aux_tasks]


    name2cls=dict(
        contrastive_hist=ContrastiveHistoryComparison,
        cont_obj_model=ContrastiveObjectModel,
        )


    aux_dict = dict()
    for aux_task in aux_tasks:
        if not aux_task in name2cls: raise RuntimeError(f"{aux_task} not supported. Only support {str(name2cls)}")
        aux_dict[aux_task] = name2cls[aux_task]
        name2cls[aux_task].update_config(config)

    return aux_dict

def load_env_setting(config, rootdir='.'):
    if not 'eval_env' in config:
        config['eval_env'] = copy.deepcopy(config['env'])

    if config['settings']['env'] == 'babyai':
        # vocab/tasks paths
        vocab_path = os.path.join(rootdir, "preloads/babyai/vocab.json")
        task_path = os.path.join(rootdir, "preloads/babyai/tasks.json")

        # dynamically load environment to use. corresponds to gym envs.
        import babyai.levels.iclr19_levels as iclr19_levels
        level = config['env']['level']
        env_class = getattr(iclr19_levels, f"Level_{level}")

    elif config['settings']['env'] == 'babyai_kitchen':
        vocab_path = os.path.join(rootdir, "preloads/babyai_kitchen/vocab.json")
        task_path = os.path.join(rootdir, "preloads/babyai_kitchen/tasks.json")
        from envs.babyai_kitchen.levelgen import KitchenLevel
        env_class = KitchenLevel
    else:
        raise RuntimeError(f"Env setting not supported: {config['settings']['env']}")

    instr_preprocessor = load_instr_preprocessor(vocab_path)
    task2idx = load_task_indices(task_path)

    # -----------------------
    # setup kwargs
    # -----------------------
    level_kwargs=config.get('level', {})
    env_kwargs=dict(
            instr_preprocessor=instr_preprocessor,
            task2idx=task2idx,
            env_class=env_class,
            level_kwargs=level_kwargs,
            )

    config['env'] = update_config(config['env'], env_kwargs)
    config['eval_env'] = update_config(config['eval_env'], env_kwargs)


    # load horizon
    env = BabyAIEnv(**config['eval_env'])
    horizon = env.horizon
    del env

    return config, instr_preprocessor, task2idx, horizon

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
    algo_kwargs['AuxClasses'] = load_aux_tasks(config)
    algo_kwargs['aux_kwargs'] = config['aux']
    algo_kwargs['train_tasks'] = train_tasks

    if config['algo'].get('joint', False):
        algo_class = R2D1AuxJoint
    else:
        algo_class = R2D1Aux

    algo = algo_class(
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

def train(config, affinity, log_dir, run_ID, name='babyai', gpu=False, parallel=True, skip_launched=True):

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
    config, instr_preprocessor, task2idx, horizon = load_env_setting(config)

    train, test = load_task_info(config)

    train_tasks = [task2idx[t] for t in train]
    eval_tasks = [task2idx[t] for t in test]


    # ======================================================
    # load algorithm (loss functions) + agent (architecture)
    # ======================================================
    algo, agent = load_algo_agent(config, horizon=horizon, train_tasks=train_tasks)




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
        env_kwargs=config['env'],
        eval_env_kwargs=config['eval_env'],
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
        eval_tasks=eval_tasks,
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
    parser.add_argument('--env',
        help='number of missions to sample (default 0 = infinity)',
        type=str,
        default="babyai_kitchen")
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
    build_and_train()
