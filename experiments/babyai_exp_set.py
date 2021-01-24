
"""
Parallel sampler version of Atari DQN.  
- Increasing the number of parallel environmnets (sampler batch_B) should improve 
  the efficiency of the forward pass for action sampling on the GPU. 
import torch.optim
- Using a larger batch size in the algorithm should improve the efficiency 
  of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
import sys
import os.path
import torch.optim

# ======================================================
# RLPYT modules
# ======================================================
from rlpyt.utils.launching.affinity import affinity_from_code


from rlpyt.samplers.parallel.gpu.sampler import GpuSampler

# from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo

# from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.r2d1 import R2D1 # algorithm

# from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
# from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1Agent


from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer
from rlpyt.utils.launching.variant import load_variant


from rlpyt.utils.logging import logger

# ======================================================
# BabyAI/Minigrid modules
# ======================================================
# import babyai.utils


# ======================================================
# Our modules
# ======================================================
# from sfgen.babyai.agents import BabyAIR2d1Agent
# from sfgen.babyai.env import BabyAIEnv
from sfgen.tools.exp_launcher import get_run_name
from sfgen.tools.variant import update_config
from sfgen.babyai.configs import configs
from experiments.babyai_exp import train

def build_and_train(slot_affinity_code, log_dir, run_ID):
    variant = load_variant(log_dir)

    global config

    if 'settings' in variant:
        settings = variant['settings']
        config_name = settings['config']
        variant_idx = settings['variant_idx']
    else:
        raise RuntimeError("settings required to get variant index")
        # config_name = 'ppo'
        # variant_idx

    config = configs[config_name]
    config = update_config(config, variant)

    affinity = affinity_from_code(slot_affinity_code)

    if "cuda_idx" in affinity:
        gpu=True
    else:
        gpu=False

    logger.set_snapshot_gap(5e5)

    experiment_title = get_run_name(log_dir)
    train(config, affinity, log_dir, run_ID,
        name=f"{experiment_title}_var{variant_idx}",
        gpu=gpu)


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
