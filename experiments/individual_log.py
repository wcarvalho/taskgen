# PYTORCH_JIT=0 python -m ipdb -c continue experiments/individual.py --cuda_idx 2 --n_parallel 32 --verbosity 0
# python -m ipdb -c continue experiments/individual.py --cuda_idx 2 --n_parallel 32 --verbosity 0

# ======================================================
# 2021.01.17 - RLDL
# replicating babyAI architecture
# ======================================================
config=dict(
    settings=dict(
        env='babyai'
    ),
    model=dict(
        batch_norm=True,
        film_batch_norm=True,
        film_pool=True,
        intrustion_policy_input=True,
        ),
    env=dict(
        level='PutNextLocal',
        ),
    level=dict(
        # num_grid=3,
        # agent_view_size=3,
        # num_dists=0,
        ),
    sampler=dict(
        eval_max_steps=int(100e3),
    ),
)

# """ ======================================================
# 2021.01.23/4 - Brain
#     - testing babyAI on own tasks
# ====================================================== """
config=dict(
    settings=dict(
        model='chaplot',
        env='babyai_kitchen',
        algorithm='r2d1',
    ),
    level=dict(
        task_kinds=['slice'],
    ),
    runner=dict(
        n_steps=5e7, # 1e6 = 1 million, 1e8 = 100 million
        log_interval_steps=1e4,
    ),
    model=dict(
        rlhead='dqn',
        batch_norm=False,
    ),
    agent=dict(
        eps_eval=0.01,
        ),
    algo=dict(
        min_steps_learn=int(0),
        eps_steps=5e6, # 5 million
        warmup_T=0,
        replay_ratio=8,
    ),
)

""" ======================================================
2021.02.01 - Brain
    - setting up replay buffer and stuff
====================================================== """
config=dict(
    settings=dict(
        model='sfgen',
        env='babyai_kitchen',
        algorithm='r2d1',
        aux='contrastive_hist',
        # gvf='goal_gvf',
    ),
    env=dict(
        task_file="test_cool_slice_01.yaml",
        ),
    level=dict(
        task_kinds=['slice'],
        num_dists=0,
        # room_size=5,
    ),
    model=dict(
        # rlhead='ppo',
        mod_compression='linear',
        batch_norm=False,
        obs_in_state=True,
        pre_mod_layer=True,
    ),
    aux=dict(
        min_trajectory=1,
        min_steps_learn=int(0),
        ),
    runner=dict(
        n_steps=5e7, # 1e6 = 1 million, 1e8 = 100 million
        log_interval_steps=2e4,
    ),
    algo=dict(
        min_steps_learn=int(1e4),
        # batch_T=20,
        # batch_B=3,
        store_rnn_state_interval=1,
        warmup_T=2,
        n_step_return=3,
        # replay_size=int(200*4),
        ),
    sampler = dict(
        batch_T=40,    # number of time-steps of data collection between optimization
        batch_B=32,    # number of parallel environents
        max_decorrelation_steps=0,
        # eval_n_envs=1,
        eval_max_trajectories=1,
    )
)

""" ======================================================
2021.02.10 - Brain
    - testing full run
====================================================== """
config=dict(
    settings=dict(
        # aux='contrastive_hist',
        aux='none',
        # gvf='goal_gvf',
    ),
    env=dict(
        task_file="cool_place_food.01.yaml",
        ),
    level=dict(
        num_dists=6,
        room_size=6,
    ),
    aux=dict(
        min_trajectory=1,
        min_steps_learn=int(0),
        ),
    model=dict(
        goal_in_state=True,
        goal_hist_depth=0,
        nonlinearity='ReLU',
        default_size=512,
        nheads=4,
        ),
    runner=dict(
        n_steps=5e7, # 1e6 = 1 million, 1e8 = 100 million
        log_interval_steps=10e4,
    ),
    algo=dict(
        min_steps_learn=int(5e3),
        replay_ratio=1,    # In the paper, more like 0.8.
        # store_rnn_state_interval=1,
        replay_size=int(5e4),
        ),
    # sampler = dict(
    #     eval_n_envs=1,                                # number of evaluation environments
    #     eval_max_trajectories=1,         # maximum # of trajectories to collect
    #     )
)


""" ======================================================
2021.02.10 - Brain
    - testing object-model
====================================================== """
config=dict(
    settings=dict(
        aux='none',
        gvf='goal_gvf',
    ),
    env=dict(
        task_file="cool_slice_place_heat_01.yaml",
        ),
    level=dict(
        num_dists=0,
        room_size=5,
    ),
    gvf=dict(
        coeff=.01,
        stop_grad=True,
        ),
    model=dict(
        default_size=1024,
        nheads=1,
        combine_state_gvf=True,
        # individual_rnn_dim=64,
        # normalize_history=True,
        # normalize_goal=True,
        ),
    runner=dict(
        n_steps=5e7, # 1e6=1 million, 1e8=100 million
        log_interval_steps=1e4,
    ),
    algo=dict(
        min_steps_learn=int(5e3),
        joint=True,
        store_rnn_state_interval=40,
        # replay_ratio=1,    # In the paper, more like 0.8.
        # replay_size=int(5e4),
        ),
    sampler=dict(
        batch_T=40,
        eval_max_trajectories=4,
        collector='wait',
        )
)