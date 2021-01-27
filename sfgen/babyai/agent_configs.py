import copy

configs = dict()

# ======================================================
# BabyAI + PPO
# ======================================================
config = dict(
    settings=dict(
        algorithm='ppo',
        ),
    agent=dict(),
    model=dict(
        dual_body=False,
        lstm_type='regular',

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
configs["ppo"] = config
config = copy.deepcopy(configs["ppo"])








# ======================================================
# SFGEN + PPO
# ======================================================
config['model']['vision_model'] = "babyai"
config['model']['lstm_type'] = 'task_gated'
config['model']['task_modulation'] = 'film'
config['model']['dual_body'] = True
config['optim']['weight_decay'] = 1e-5
configs["sfgen_ppo"] = config
config = copy.deepcopy(configs["sfgen_ppo"])







# ======================================================
# SFGEN + R2D1
# ======================================================
# config['algoname'] = 'r2d1'
config['algo']=dict(
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

config['sampler'] = dict(
        batch_T=64,    # number of time-steps of data collection between optimization
        batch_B=32,    # number of parallel environents
        max_decorrelation_steps=1000,    # used to get random actions into buffer
        eval_n_envs=32,                                # number of evaluation environments
        eval_max_steps=int(1e5),            # number of TOTAL steps of evaluation
        eval_max_trajectories=100,         # maximum # of trajectories to collect
    )

config['settings']['algorithm'] = 'r2d1'
config['model']['dueling'] = True
config['env']['reward_scale'] = 1
# config['eval_env']['reward_scale'] = 1

configs["sfgen_r2d1"] = config
config = copy.deepcopy(configs["sfgen_r2d1"])









