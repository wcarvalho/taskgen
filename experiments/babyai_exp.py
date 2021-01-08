
"""
Parallel sampler version of Atari DQN.  
- Increasing the number of parallel environmnets (sampler batch_B) should improve 
  the efficiency of the forward pass for action sampling on the GPU. 
- Using a larger batch size in the algorithm should improve the efficiency 
  of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""

import os
# ======================================================
# RLPYT modules
# ======================================================
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector

# from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo

# from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.r2d1 import R2D1 # algorithm

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
from sfgen.agents.babyai_agent import BabyAIR2d1Agent
from sfgen.envs.babyai_env import BabyAIEnv
from sfgen.configs.babyai_r2d1 import configs


def build_and_train(
    level="pong",
    run_ID=0,
    cuda_idx=None,
    n_parallel=2,
    use_pixels=False,
    log_dir="logs",
    n_steps=1e6,
    log_interval_steps=2e5,
    gpu_sampler=False,
    debug=False,
    num_missions=0,
    snapshot_gap=10
    ):

    instr_preprocessor = babyai.utils.format.InstructionsPreprocessor(model_name="babyai")

    path = instr_preprocessor.vocab.path
    if not os.path.exists(path):
        raise RuntimeError(f"Please create vocab and put in {path}")
    else:
        print(f"Successfully loaded {path}")


    config = configs['r2d1']
    instr_preprocessor = babyai.utils.format.InstructionsPreprocessor(model_name="babyai")
    config['env'].update(
        dict(
            instr_preprocessor=instr_preprocessor,
            num_missions=num_missions,
            ))

    if cuda_idx is not None and torch.cuda.is_available():
        sampler_class = GpuSampler
    else:
        sampler_class = SerialSampler


    algo = R2D1(
        ReplayBufferCls=PrioritizedSequenceReplayBuffer,
        **config["algo"])  # Run with defaults.
    agent = BabyAIR2d1Agent(model_kwargs=config['model'])
    sampler = sampler_class(
        EnvCls=BabyAIEnv,
        Collecto
        env_kwargs=config['env'],
        eval_env_kwargs=config['env'],
        **config["sampler"]  # More parallel environments for batched forward-pass.
    )



    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        **config["runner"],
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel))),
    )

    name = f"r2d1_{level}"
    log_dir = f"data/logs/{log_dir}/{name}"
    
    logger.set_snapshot_gap(snapshot_gap)
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
    parser.add_argument('--use_pixels', help='whether to learn from raw pixels', type=int, default=1)
    parser.add_argument('--gpu_sampler', help='whether to sample on GPU', type=int, default=1)
    parser.add_argument('--debug', help='whether to debug', type=int, default=0)
    parser.add_argument('--n_steps', help='number of environment steps (default=1 million)', type=int, default=2e6)
    parser.add_argument('--num_missions', help='number of missions to sample (default 0 = infinity)', type=int, default=0)
    parser.add_argument('--log_interval_steps', help='Number of environment steps between logging to csv/tensorboard/etc (default=100 thousand)', type=int, default=1e5)
    parser.add_argument('--snapshot-gap', help='how', type=int, default=1e5)
    
    args = parser.parse_args()
    build_and_train(**vars(args))
