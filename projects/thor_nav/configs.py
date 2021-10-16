from utils.variant import update_config
import copy

model_configs = dict()
algorithm_configs = dict()
env_configs = dict()

configs=dict(
    model=model_configs,
    env=env_configs,
    algorithm=algorithm_configs,
    )

defaults=dict(
    model='resnet_lstm',
    env='thor_nav',
    algorithm='ppo',
)

# ======================================================
# Model configs
# ======================================================

model_config = dict(
    settings=dict(
        model='resnet_lstm',
        ),
    agent=dict(),
    model=dict(
        fc_sizes=512,  # Between conv and lstm.
        lstm_size=512,
        head_size=256,
        task_size=256,
        action_size=64,
    ),
)
model_configs["resnet_lstm"] = model_config

# ======================================================
# Algorithm configs
# ======================================================

# -----------------------
# PPO
# -----------------------
algorithm_config = dict(
    settings=dict(
        algorithm='ppo',
        ),
    optim=dict(
        eps=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    ),
    runner=dict(
        n_steps=5e7, # 1e6 = 1 million, 1e8 = 100 million
        log_interval_steps=1e6,
    ),
    model=dict(rlhead='ppo'),
    algo=dict(
        epochs=4,
        discount=0.99,
        gae_lambda=0.99,
        learning_rate=5e-5,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=0.5,
        ratio_clip=0.2,
        linear_lr_schedule=False,
    ),
    env = dict(reward_scale=1),
    sampler = dict(
        batch_T=128,    # number of time-steps of data collection between optimization
        batch_B=10,    # number of parallel environents
        max_decorrelation_steps=1000,    # used to get random actions into buffer
        eval_n_envs=2,                                # number of evaluation envs
        eval_max_steps=int(5e5),            # number of TOTAL steps of evaluation
        eval_max_trajectories=400,         # maximum # of trajectories to collect
    ),
)
algorithm_configs["ppo"] = algorithm_config
# copy
algorithm_config = copy.deepcopy(algorithm_configs["ppo"])


# -----------------------
# R2D1 (DQN)
# -----------------------
algorithm_config.update(dict(
    settings=dict(
        algorithm='r2d1',
    ),
    agent=dict(
        eps_final=0.01,
        eps_eval=0.1,
        ),
    algo=dict(
        eps_steps=1e7, # 10 million
        discount=0.99,
        batch_T=40,
        batch_B=32,    # In the paper, 64.
        warmup_T=0,
        store_rnn_state_interval=40,
        replay_ratio=4,    # In the paper, more like 0.8.
        learning_rate=5e-5,
        clip_grad_norm=80.,    # 80 (Steven.)
        min_steps_learn=int(1e5),
        replay_size=int(5e5),
        double_dqn=True,
        prioritized_replay=True,
        n_step_return=5,
        pri_alpha=0.9,    # Fixed on 20190813
        pri_beta_init=0.6,    # I think had these backwards before.
        pri_beta_final=0.6,
        input_priority_shift=2,    # Added 20190826 (used to default to 1)
        joint=True,
    ),
    sampler=dict(
            batch_T=40,    # number of time-steps of data collection between optimization
            batch_B=10,    # number of parallel environents
            max_decorrelation_steps=1000,    # used to get random actions into buffer
            eval_n_envs=2,                                # number of evaluation envs
            eval_max_steps=int(200*10),            # number of TOTAL steps of evaluation
            eval_max_trajectories=100,         # maximum # of trajectories to collect
        ),
    env=dict(
        reward_scale=1,
        )

))
algorithm_configs["r2d1"] = algorithm_config
algorithm_config = copy.deepcopy(algorithm_configs["r2d1"])





# ======================================================
# Environment configs
# ======================================================

# -----------------------
# BabyAI env
# -----------------------
env_config = dict(
    settings=dict(
        env='thor_nav',
    ),
    env=dict(
        timestep_penalty=-0.004,
        success_distance=2.0,
        actions=["MoveAhead", "MoveBack", "RotateRight", "RotateLeft", "LookUp", "LookDown"],
        controller_kwargs=dict(
          quality='MediumCloseFitShadows', # Alfred
          ),
        tasks_in_floorplan=50,
        max_steps=200,
        init_kwargs=dict(
          # camera properties
          width=300,
          height=300,
          fieldOfView=90,
          # step sizes
          gridSize=0.25,
          visibility_distance=1.5,
          rotateStepDegrees=90,
          rotateHorizonDegrees=30,
          )
    )
)
env_configs["thor_nav"] = env_config
