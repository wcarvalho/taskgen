from utils.variant import update_config
import copy

model_configs = dict()
algorithm_configs = dict()
env_configs = dict()

configs=dict(
    model_configs=model_configs,
    env_configs=env_configs,
    algorithm_configs=algorithm_configs,
    )

defaults=dict(
    model='babyai',
    env='babyai',
    algorithm='r2d1',
)

# ======================================================
# Model configs
# ======================================================

model_config = dict(
    settings=dict(
        model='babyai',
        ),
    agent=dict(),
    model=dict(
        dual_body=False,
        lstm_type='regular',
        rlhead='ppo',
        #vision model
        vision_model="babyai",
        use_maxpool=False,
        batch_norm=True,
        use_pixels=True,
        use_bow=False,
        # language
        lang_model='bigru',
        text_embed_size=128,
        # obs modulation
        task_modulation='film',
        film_bias=True,
        film_batch_norm=True,
        film_residual=True,
        film_pool=True,
        # policy
        intrustion_policy_input=True,
        lstm_size=128,
        head_size=64,
        fc_size=0, # no intermediate layer before LSTM
    ),
)
model_configs["babyai"] = model_config

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
    env = dict(reward_scale=20),
    sampler = dict(
        batch_T=40,    # number of time-steps of data collection between optimization
        batch_B=64,    # number of parallel environents
        max_decorrelation_steps=1000,    # used to get random actions into buffer
        eval_n_envs=32,                                # number of evaluation envs
        eval_max_steps=int(5e5),            # number of TOTAL steps of evaluation
        eval_max_trajectories=400,         # maximum # of trajectories to collect
    ),
    model=dict(
        dueling=False,
        rlhead='ppo',
    )

)
algorithm_configs["ppo"] = algorithm_config
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
            batch_B=32,    # number of parallel environents
            max_decorrelation_steps=1000,    # used to get random actions into buffer
            eval_n_envs=32,                                # number of evaluation envs
            eval_max_steps=int(5e5),            # number of TOTAL steps of evaluation
            eval_max_trajectories=400,         # maximum # of trajectories to collect
        ),
    model=dict(
        dueling=False,
        rlhead='dqn',
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
        env='babyai',
    ),
    env=dict(
        level="GoToLocal",
        use_pixels=True,
        num_missions=0,
        tile_size=8,
        timestep_penalty=-0.004,
    )
)
env_configs["babyai"] = env_config
