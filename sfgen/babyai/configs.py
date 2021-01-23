import copy

configs = dict()

# ======================================================
# Default BabyAI settings
# ======================================================
config = dict(
  settings=dict(algorithm='ppo_babyai'),
  agent=dict(),
  model=dict(
    rlalgorithm='ppo',
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
  env=dict(
    level="GoToLocal",
    use_pixels=True,
    num_missions=0,
    reward_scale=20,
  ),
  # eval_env=dict(
  #   level="GoToLocal",
  #   use_pixels=True,
  #   num_missions=0,
  #   reward_scale=20,
  # ),
  runner=dict(
    n_steps=5e7, # 1e6 = 1 million, 1e8 = 100 million
    log_interval_steps=2.5e5,
  ),
  level=dict(
    )
)


# ======================================================
# PPO
# ======================================================
config['algo']=dict(
    epochs=4,
    discount=0.99,
    gae_lambda=0.99,
    learning_rate=1e-4,
    value_loss_coeff=0.5,
    entropy_loss_coeff=0.01,
    clip_grad_norm=0.5,
    ratio_clip=0.2,
    linear_lr_schedule=False,
)

config['sampler'] = dict(
    batch_T=40,  # number of time-steps of data collection between optimization
    batch_B=64,  # number of parallel environents
    max_decorrelation_steps=1000,  # used to get random actions into buffer
    eval_n_envs=32,                # number of evaluation environments
    eval_max_steps=int(1e5),      # number of TOTAL steps of evaluation
    eval_max_trajectories=100,     # maximum # of trajectories to collect
  )

configs["ppo_babyai"] = config
config = copy.deepcopy(configs["ppo_babyai"])

# ======================================================
# Our defaults
# ======================================================
config['model']['vision_model'] = "babyai"
config['model']['lstm_type'] = 'task_gated'
config['model']['task_modulation'] = 'film'
config['model']['dual_body'] = True
config['optim']['weight_decay'] = 1e-5
configs["ppo"] = config
config = copy.deepcopy(configs["ppo"])


# ======================================================
# R2D1
# ======================================================
# config['algoname'] = 'r2d1'
config['algo']=dict(
    discount=0.99,
    batch_T=40,
    batch_B=32,  # In the paper, 64.
    warmup_T=40,
    store_rnn_state_interval=40,
    replay_ratio=4,  # In the paper, more like 0.8.
    learning_rate=1e-4,
    clip_grad_norm=80.,  # 80 (Steven.)
    min_steps_learn=int(1e5),
    double_dqn=True,
    prioritized_replay=True,
    n_step_return=5,
    pri_alpha=0.9,  # Fixed on 20190813
    pri_beta_init=0.6,  # I think had these backwards before.
    pri_beta_final=0.6,
    input_priority_shift=2,  # Added 20190826 (used to default to 1)
  )

config['sampler'] = dict(
    batch_T=64,  # number of time-steps of data collection between optimization
    batch_B=32,  # number of parallel environents
    max_decorrelation_steps=1000,  # used to get random actions into buffer
    eval_n_envs=32,                # number of evaluation environments
    eval_max_steps=int(1e5),      # number of TOTAL steps of evaluation
    eval_max_trajectories=100,     # maximum # of trajectories to collect
  )

config['model']['dueling'] = True
config['model']['rlalgorithm'] = 'dqn'
config['env']['reward_scale'] = 1
# config['eval_env']['reward_scale'] = 1

configs["r2d1"] = config




















config = copy.deepcopy(configs["r2d1"])
config["algo"]["replay_size"] = int(4e6)  # Even bigger is better (Steven).
config["algo"]["batch_B"] = 64  # Not sure will fit.
config["algo"]["replay_ratio"] = 1
# config["algo"]["eps_final"] = 0.1  # Now in agent.
# config["algo"]["eps_final_min"] = 0.0005
config["agent"]["eps_final"] = 0.1  # (Steven: 0.4 - 0.4 ** 8 =0.00065)
config["agent"]["eps_final_min"] = 0.0005  # (Steven: approx log space but doesn't matter)
config["runner"]["n_steps"] = 20e9
config["runner"]["log_interval_steps"] = 10e6
config["sampler"]["batch_T"] = 40  # = warmup_T = store_rnn_interval; new traj at boundary.
config["sampler"]["batch_B"] = 192  # to make one update per sample batch.
config["sampler"]["eval_n_envs"] = 6  # 6 cpus, 6 * 32 = 192, for pabti.
config["sampler"]["eval_max_steps"] = int(28e3 * 6)
config["env"]["episodic_lives"] = False  # Good effects some games (Steven).
configs["r2d1_long"] = config
