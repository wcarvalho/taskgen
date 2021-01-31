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
        env='babyai_kitchen'
    ),
    level=dict(
        task_kinds=['heat'],
    )
)