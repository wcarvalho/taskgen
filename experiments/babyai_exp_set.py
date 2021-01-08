
"""
Parallel sampler version of Atari DQN.  
- Increasing the number of parallel environmnets (sampler batch_B) should improve 
  the efficiency of the forward pass for action sampling on the GPU. 
- Using a larger batch size in the algorithm should improve the efficiency 
  of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
import sys
import os.path


# ======================================================
# RLPYT modules
# ======================================================
from rlpyt.utils.launching.affinity import affinity_from_code


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
from rlpyt.utils.launching.variant import load_variant, update_config



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


def build_and_train(slot_affinity_code, log_dir, run_ID):
    global config
    config = configs['r2d1']
    variant = load_variant(log_dir)
    config = update_config(config, variant)


    # I BET you loading this is the problem...
    instr_preprocessor = babyai.utils.format.InstructionsPreprocessor(model_name="babyai")

    path = instr_preprocessor.vocab.path
    if not os.path.exists(path):
        raise RuntimeError(f"Please create vocab and put in {path}")
    else:
        print(f"Successfully loaded {path}")

    config['env'].update(
        dict(instr_preprocessor=instr_preprocessor))
    # config['eval_env'] = config['env']



    algo = R2D1(
        ReplayBufferCls=PrioritizedSequenceReplayBuffer,
        **config["algo"])  # Run with defaults.
    agent = BabyAIR2d1Agent(model_kwargs=config['model'])
    sampler = GpuSampler(
        EnvCls=BabyAIEnv,
        # CollectorCls=GpuWaitResetCollector,
        env_kwargs=config['env'],
        eval_env_kwargs=config['env'],
        **config["sampler"]  # More parallel environments for batched forward-pass.
    )



    affinity = affinity_from_code(slot_affinity_code)
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )



    name = "r2d1"
    # log_dir = f"{log_dir}"
    with logger_context(
        log_dir,
        run_ID,
        name,
        config,
        override_prefix=True,
        use_summary_writer=True,
        ):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
