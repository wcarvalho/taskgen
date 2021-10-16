config=dict(
    env=dict(
        level="PutNextLocal",
        ),
    runner=dict(
        n_steps=5e7, # 1e6=1 million, 1e8=100 million
        log_interval_steps=1e4,
    ),
    algo=dict(
        min_steps_learn=int(5e3),
        ),
)