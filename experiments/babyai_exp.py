
"""
Parallel sampler version of Atari DQN.  
- Increasing the number of parallel environmnets (sampler batch_B) should improve 
  the efficiency of the forward pass for action sampling on the GPU. 
- Using a larger batch size in the algorithm should improve the efficiency 
  of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
import multiprocessing
import os
import json
import torch.cuda

try:
    import wandb
    WANDB_AVAILABLE=True
except Exception as e:
    WANDB_AVAILABLE=False

# ======================================================
# RLPYT modules
# ======================================================
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler

from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo

from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.r2d1 import R2D1 # algorithm
from rlpyt.algos.pg.ppo import PPO # algorithm

from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1Agent

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
from sfgen.babyai.agents import BabyAIR2d1Agent, BabyAIPPOAgent
from sfgen.babyai.babyai_model import BabyAIRLModel
from sfgen.babyai.sfgen_model import SFGenModel
from sfgen.babyai.configs import algorithm_configs, model_configs, env_configs, aux_configs
from sfgen.babyai.env import BabyAIEnv
from sfgen.tools.variant import update_config
import experiments.individual_log as log

def load_config(settings,
    default_env='babyai_kitchen',
    default_model='sfgen',
    default_algorithm='ppo',
    default_aux='none',
    ):
    env = settings.get("env", default_env)
    model = settings.get("model", default_model)
    algorithm = settings.get("algorithm", default_algorithm)
    aux = settings.get("aux", default_aux)

    config = env_configs[env]
    config = update_config(config, model_configs[model])
    config = update_config(config, algorithm_configs[algorithm])
    config = update_config(config, aux_configs[aux])

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
    algorithm='ppo',
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


def load_instr_preprocessor(path="models/babyai/vocab.json"):
    instr_preprocessor = babyai.utils.format.InstructionsPreprocessor(path=path)

    path = instr_preprocessor.vocab.path
    if not os.path.exists(path):
        raise RuntimeError(f"Please create vocab and put in {path}")
    else:
        print(f"Successfully loaded {path}")

    return instr_preprocessor

def load_task_indices(path="models/babyai/tasks.json"):
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


def train(config, affinity, log_dir, run_ID, name='babyai', gpu=False, parallel=True, wandb=False):

    # ======================================================
    # load environment settings
    # ======================================================
    if config['settings']['env'] == 'babyai':
        # vocab/tasks paths
        vocab_path = "models/babyai/vocab.json"
        task_path = "models/babyai/tasks.json"

        # dynamically load environment to use. corresponds to gym environments.
        import babyai.levels.iclr19_levels as iclr19_levels
        level = config['env']['level']
        env_class = getattr(iclr19_levels, f"Level_{level}")
    elif config['settings']['env'] == 'babyai_kitchen':
        vocab_path = "models/babyai_kitchen/vocab.json"
        task_path = "models/babyai_kitchen/tasks.json"
        from sfgen.babyai_kitchen.levelgen import KitchenLevel
        env_class = KitchenLevel
    else:
        raise RuntimeError(f"Env setting not supported: {config['settings']['env']}")

    instr_preprocessor = load_instr_preprocessor(vocab_path)
    task2idx = load_task_indices(task_path)

    config['env'].update(
        dict(
            instr_preprocessor=instr_preprocessor,
            task2idx=task2idx,
            env_class=env_class,
            ),
        level_kwargs=config.get('level', {}),
        )



    # ======================================================
    # load sampler
    # ======================================================
    if gpu:
        sampler_class = GpuSampler
    else:
        if parallel:
            sampler_class = CpuSampler
        else:
            sampler_class = SerialSampler
    sampler = sampler_class(
        EnvCls=BabyAIEnv,
        env_kwargs=config['env'],
        eval_env_kwargs=config['env'],
        **config["sampler"]  # More parallel environments for batched forward-pass.
    )



    # ======================================================
    # Load Agent
    # ======================================================
    # -----------------------
    # model
    # -----------------------
    if config['settings']['model'] in ['babyai', 'chaplot']:
        ModelCls = BabyAIRLModel
    elif config['settings']['model'] in ['sfgen']:
        ModelCls = SFGenModel
    else: raise NotImplementedError
    # -----------------------
    # auxilliary task
    # -----------------------
    if config['settings']['aux'] in ['contrastive_hist']:
        config["algo"]['AuxCls'] = ContrastiveHistoryComparison
    elif config['settings']['aux'] == 'none':
        pass
    else: raise NotImplementedError

    # -----------------------
    # algorithm + agent
    # -----------------------
    if config['settings']['algorithm'] in ['r2d1']:
        assert config['model']['rlhead'] in ['dqn', 'successor_dqn']
        if config['settings']['aux'] != 'none':
            algo_class = R2D1Aux
        else:
            algo_class = R2D1
        algo = algo_class(
            ReplayBufferCls=PrioritizedSequenceReplayBuffer,
            optim_kwargs=config['optim'],
            **config["algo"]
            )  # Run with defaults.
        agent = BabyAIR2d1Agent(
            **config['agent'],
            ModelCls=ModelCls,
            model_kwargs=config['model'],
            )
    elif config['settings']['algorithm'] in ['ppo']:
        assert config['model']['rlhead'] in ['ppo']
        if config['settings']['aux'] != 'none':
            algo_class = PPOAux
        else:
            algo_class = PPO
        algo = algo_class(
            optim_kwargs=config['optim'],
            **config["algo"]
            )  # Run with defaults.
        agent = BabyAIPPOAgent(
            ModelCls=ModelCls,
            model_kwargs=config['model'],
            **config['agent'],
            )
    else:
        raise NotImplementedError(f"Algo: {config['settings']['algorithm']}")


    # ======================================================
    # Load runner + train
    # ======================================================
    if wandb and WANDB_AVAILABLE:
        from sfgen.tools.runners import MinibatchRlEvalWandb
        runner_class = MinibatchRlEvalWandb
        wandb.init(
            project="sfgen",
            entity="wcarvalho92",
            group=name,
            config=config
            )
    else:
        from rlpyt.runners.minibatch_rl import MinibatchRlEval
        runner_class = MinibatchRlEval

    runner = runner_class(
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
    build_and_train(**vars(args))
