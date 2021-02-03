# python -m ipdb -c continue experiments/babyai_exp.py --cuda_idx 3 --n_parallel 32 --verbosity 0

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
        # room_size=12,
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
        model='chaplot',
        env='babyai_kitchen',
        algorithm='ppo',
        aux='contrastive_hist',
    ),
    level=dict(
        task_kinds=['slice'],
    ),
    model=dict(
        rlhead='ppo',
        batch_norm=False,
    )
)