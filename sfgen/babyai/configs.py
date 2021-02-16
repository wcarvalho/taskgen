from sfgen.tools.variant import update_config
import copy

# ======================================================
# Model configs
# ======================================================
model_configs = dict()

# -----------------------
# BabyAI
# -----------------------
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
model_config = copy.deepcopy(model_configs["babyai"])


# -----------------------
# chaplot
# -----------------------
model_config['settings']['model'] = 'chaplot'
model_config['model']['task_modulation'] = 'chaplot'
model_config['model']['fc_size'] = 128
model_configs["chaplot"] = model_config
model_config = copy.deepcopy(model_configs["chaplot"])



# -----------------------
# SFGEN
# -----------------------
model_config = update_config(model_config, dict(
    settings=dict(
        model='sfgen',
        ),
    model=dict(
        mod_function='sigmoid',
        mod_compression='linear',
        goal_tracking='lstm',
        # lstm_size=128,
        # head_size=128, 
        # obs_fc_size=128,
        # gvf_size=256,
        batch_norm=True,
        default_size=512,
        obs_in_state=False,
        goal_use_history=False,
        dueling=False,
        rlhead='dqn',
        )
))

model_configs["sfgen"] = model_config
model_config = copy.deepcopy(model_configs["sfgen"])

























# ======================================================
# Algorithm configs
# ======================================================
algorithm_configs = dict()

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
        eval_n_envs=32,                                # number of evaluation environments
        eval_max_steps=int(5e5),            # number of TOTAL steps of evaluation
        eval_max_trajectories=200,         # maximum # of trajectories to collect
    ),
    model=dict(
        dueling=False,
        rlhead='ppo',
    )

)
algorithm_configs["ppo"] = algorithm_config
algorithm_config = copy.deepcopy(algorithm_configs["ppo"])



# -----------------------
# R2D1
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
        replay_size=int(1e6),
        double_dqn=True,
        prioritized_replay=True,
        n_step_return=5,
        pri_alpha=0.9,    # Fixed on 20190813
        pri_beta_init=0.6,    # I think had these backwards before.
        pri_beta_final=0.6,
        input_priority_shift=2,    # Added 20190826 (used to default to 1)
    ),
    sampler=dict(
            batch_T=64,    # number of time-steps of data collection between optimization
            batch_B=32,    # number of parallel environents
            max_decorrelation_steps=1000,    # used to get random actions into buffer
            eval_n_envs=32,                                # number of evaluation environments
            eval_max_steps=int(5e5),            # number of TOTAL steps of evaluation
            eval_max_trajectories=200,         # maximum # of trajectories to collect
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
# Auxilliary Task
# ======================================================
aux_configs = dict()

aux_config = dict(
    settings=dict(aux='none'),
    aux=dict(),
)
aux_configs["none"] = aux_config
aux_config = copy.deepcopy(aux_configs["none"])


# -----------------------
# Contrastive History Estimation
# -----------------------
aux_config = dict(
    settings=dict(
        aux='contrastive_hist',
        ),
    aux=dict(
        temperature=0.1,
        symmetric=True,
        num_timesteps=10,
        min_trajectory=1,
        epochs=5,
        min_steps_learn=int(1e5),
        ),
    model=dict(
        normalize_history=True,
        ),
    algo=dict(
        buffer_type='multitask',
        warmup_T=0,
        store_rnn_state_interval=1,
        ),
)
aux_configs["contrastive_hist"] = aux_config
aux_config = copy.deepcopy(aux_configs["contrastive_hist"])






# ======================================================
# GVF
# ======================================================
gvf_configs = dict()

gvf_config = dict(
    settings=dict(gvf='none'),
    gvf=dict(),
)
gvf_configs["none"] = gvf_config
gvf_config = copy.deepcopy(gvf_configs["none"])


# -----------------------
# Contrastive History Estimation
# -----------------------
gvf_config = dict(
    settings=dict(
        gvf='goal_gvf',
        ),
    gvf=dict(),
)
gvf_configs["goal_gvf"] = gvf_config
gvf_config = copy.deepcopy(gvf_configs["goal_gvf"])







































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
        strict_task_idx_loading=False,
        tile_size=8,
        timestep_penalty=-0.004,
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
        tile_size=12,
    ),
    env=dict(
        strict_task_idx_loading=True,
        use_pixels=True,
        num_missions=0,
        tile_size=12,
        timestep_penalty=-0.004,
    ),
))
env_configs["babyai_kitchen"] = env_config
