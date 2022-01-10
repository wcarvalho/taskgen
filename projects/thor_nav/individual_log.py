# ======================================================
# Local Testing (Toy Run)
# ======================================================
config=dict(
    env=dict(verbosity=1),
    runner=dict(
        n_steps=100e3, # 100k
        log_interval_steps=1e3, # 1K
    ),
    algo=dict(),
    sampler=dict(
            batch_B=1,
            eval_n_envs=1,
            )
)

# ======================================================
# Server, Testing (Toy Run)
# ======================================================
config=dict(
    env=dict(
      verbosity=1,
      max_steps=200,
      ),
    runner=dict(
        n_steps=100e3, # 100k
        log_interval_steps=1e3, # 1K
    ),
    algo=dict(),
    model=dict(
      out_conv=32,
      ),
    sampler=dict(
            batch_B=8, # 8 train processes
            eval_n_envs=2, # 2 eval processes
            eval_max_trajectories=2, # 1 eval per process
            )
)
