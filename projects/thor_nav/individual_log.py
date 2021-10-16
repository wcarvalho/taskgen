# ======================================================
# Local Testing
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
# Server, Testing
# ======================================================
config=dict(
    env=dict(
      verbosity=1,
      max_steps=50,
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

# # ======================================================
# # Server, Single Run
# # ======================================================
# config=dict(
#     env=dict(verbosity=0),
#     runner=dict(
#         n_steps=5e7, # 1e6=1 million, 1e8=100 million
#         log_interval_steps=50e3, # 100K
#     ),
#     algo=dict(),
#     sampler=dict(
#             batch_B=10,
#             eval_n_envs=2,
#             )
# )