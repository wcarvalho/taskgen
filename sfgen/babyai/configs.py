import copy

# ======================================================
# Agent configs
# ======================================================
agent_configs = dict()

# -----------------------
# BabyAI + PPO
# -----------------------
agent_config = dict(
    settings=dict(
        algorithm='ppo',
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
        fc_size=0,
    ),
    optim=dict(
        eps=1e-5,
        betas=(0.9, 0.999),
    ),
    runner=dict(
        n_steps=5e7, # 1e6 = 1 million, 1e8 = 100 million
        log_interval_steps=2.5e5,
    ),
    level=dict(
    ),
    algo=dict(
        epochs=4,
        discount=0.99,
        gae_lambda=0.99,
        learning_rate=1e-4,
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
        eval_n_envs=32,                                # number of evaluation environments
        eval_max_steps=int(1e5),            # number of TOTAL steps of evaluation
        eval_max_trajectories=100,         # maximum # of trajectories to collect
    )
)
agent_configs["ppo"] = agent_config
agent_config = copy.deepcopy(agent_configs["ppo"])








# -----------------------
# SFGEN + PPO
# -----------------------
agent_config['model']['vision_model'] = "babyai"
agent_config['model']['lstm_type'] = 'task_gated'
agent_config['model']['task_modulation'] = 'film'
agent_config['model']['dual_body'] = True
agent_config['optim']['weight_decay'] = 1e-5
agent_configs["sfgen_ppo"] = agent_config
agent_config = copy.deepcopy(agent_configs["sfgen_ppo"])







# -----------------------
# SFGEN + R2D1
# -----------------------
# agent_config['algoname'] = 'r2d1'
agent_config['algo']=dict(
        discount=0.99,
        batch_T=40,
        batch_B=32,    # In the paper, 64.
        warmup_T=40,
        store_rnn_state_interval=40,
        replay_ratio=4,    # In the paper, more like 0.8.
        learning_rate=1e-4,
        clip_grad_norm=80.,    # 80 (Steven.)
        min_steps_learn=int(1e5),
        double_dqn=True,
        prioritized_replay=True,
        n_step_return=5,
        pri_alpha=0.9,    # Fixed on 20190813
        pri_beta_init=0.6,    # I think had these backwards before.
        pri_beta_final=0.6,
        input_priority_shift=2,    # Added 20190826 (used to default to 1)
    )

agent_config['sampler'] = dict(
        batch_T=64,    # number of time-steps of data collection between optimization
        batch_B=32,    # number of parallel environents
        max_decorrelation_steps=1000,    # used to get random actions into buffer
        eval_n_envs=32,                                # number of evaluation environments
        eval_max_steps=int(1e5),            # number of TOTAL steps of evaluation
        eval_max_trajectories=100,         # maximum # of trajectories to collect
    )

agent_config['settings']['algorithm'] = 'r2d1'
agent_config['model']['dueling'] = False
agent_config['model']['rlhead'] = 'dqn'
agent_config['env']['reward_scale'] = 1
# agent_config['eval_env']['reward_scale'] = 1

agent_configs["sfgen_r2d1"] = agent_config
agent_config = copy.deepcopy(agent_configs["sfgen_r2d1"])























# ======================================================
# Environment configs
# ======================================================
env_configs = dict()

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
        strict_task_idx_loading=True,
    )
)
env_configs["babyai"] = env_config

# -----------------------
# Kitchen env
# -----------------------
env_config = copy.deepcopy(env_configs["babyai"])
env_config.update(dict(
    settings=dict(
        env='babyai_kitchen',
    ),
    level=dict(
        task_kinds=['slice', 'cool'],
        actions = ['left', 'right', 'forward', 'pickup_container', 'pickup_contents', 'place', 'toggle', 'slice'],
        room_size=8,
        agent_view_size=7,
        num_dists=5,
        random_object_state=False,
        use_time_limit=True,
        ),
    )
)
env_configs["babyai_kitchen"] = env_config
