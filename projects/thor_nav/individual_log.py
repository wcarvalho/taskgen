config=dict(
    env=dict(verbosity=1),
    runner=dict(
        n_steps=5e7, # 1e6=1 million, 1e8=100 million
        log_interval_steps=1e4,
    ),
    algo=dict(),
    sampler=dict(
            batch_B=1,
            eval_n_envs=1,
            )
)