
"""
Parallel sampler version of Atari DQN.  
- Increasing the number of parallel environmnets (sampler batch_B) should improve 
  the efficiency of the forward pass for action sampling on the GPU. 
- Using a larger batch size in the algorithm should improve the efficiency 
  of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
import os
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

# from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo

# from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.r2d1 import R2D1 # algorithm
from rlpyt.algos.pg.ppo import PPO # algorithm

# from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
# from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1Agent

from rlpyt.runners.minibatch_rl import MinibatchRlEval
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
from sfgen.tools.variant import update_config
from sfgen.tools.runners import MinibatchRlEvalWandb

from sfgen.babyai.agents import BabyAIR2d1Agent, BabyAIPPOAgent
from sfgen.babyai.env import BabyAIEnv
from sfgen.babyai.configs import configs

import experiments.individual_log as log

def build_and_train(
    level="pong",
    run_ID=0,
    cuda_idx=None,
    n_parallel=2,
    input_type='pixels',
    log_dir="logs",
    n_steps=1e6,
    log_interval_steps=2e5,
    debug=False,
    num_missions=0,
    snapshot_gap=10,
    config='dqn',
    ):

    config_name = config
    config = configs[config]
    config['env'].update(
        dict(
            # instr_preprocessor=instr_preprocessor,
            num_missions=num_missions,
            use_pixels=input_type=="pixels",
            ))
    config = update_config(config, log.config)

    gpu=cuda_idx is not None and torch.cuda.is_available()
    print("="*20)
    print(f"Using GPU: {gpu}")
    print("="*20)

    affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))

    name = f"{config_name}_{level}"
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


def load_algo_agent(config, algo_kwargs={}, agent_kwargs={}):
    if config['model']['rlalgorithm'] in ['dqn', 'r2d1']:
        algo = R2D1(
            # ReplayBufferCls=PrioritizedSequenceReplayBuffer,
            optim_kwargs=config['optim'],
            **algo_kwargs,
            **config["algo"]
            )  # Run with defaults.
        agent = BabyAIR2d1Agent(
            **config['agent'],
            model_kwargs=config['model'],
            **agent_kwargs
            )
    elif config['model']['rlalgorithm']=='ppo':
        algo = PPO(
            # ReplayBufferCls=PrioritizedSequenceReplayBuffer,
            optim_kwargs=config['optim'],
            **algo_kwargs,
            **config["algo"]
            )  # Run with defaults.
        agent = BabyAIPPOAgent(
            model_kwargs=config['model'],
            **config['agent'],
            **agent_kwargs,
            )
    else:
        raise NotImplemented(f"Algo: {config['model']['rlalgorithm']}")
    return algo, agent


def train(config, affinity, log_dir, run_ID, name='babyai', gpu=False, parallel=True):

    # ======================================================
    # load instruction processor
    # ======================================================
    instr_preprocessor = load_instr_preprocessor()
    config['env'].update(
        dict(instr_preprocessor=instr_preprocessor),
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
    algo, agent = load_algo_agent(config)

    # ======================================================
    # Load runner
    # ======================================================

    if WANDB_AVAILABLE:
        runner_class = MinibatchRlEvalWandb
        # wandb.login()

        wandb.init(
            project="sfgen",
            entity="wcarvalho92",
            group=name,
            config=config
            )
    else:
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
    parser.add_argument('--level', help='BabyAI level', default='GoToRedBall')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--log_dir', type=str, default='babyai')
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=32)
    parser.add_argument('--input-type', help='what to learn from: original tensor, BOW representation, or pixels', choices=['pixels', 'original', 'bow'], type=str, default='pixels')
    parser.add_argument('--debug', help='whether to debug', type=int, default=0)
    parser.add_argument('--n_steps', help='number of environment steps (default=1 million)', type=int, default=2e6)
    parser.add_argument('--num_missions', help='number of missions to sample (default 0 = infinity)', type=int, default=0)
    parser.add_argument('--log_interval_steps', help='Number of environment steps between logging to csv/tensorboard/etc (default=100 thousand)', type=int, default=1e5)
    parser.add_argument('--snapshot-gap', help='how', type=int, default=5)
    parser.add_argument('--config', help='which config to load', type=str, default='ppo_babyai')

    args = parser.parse_args()
    build_and_train(**vars(args))
