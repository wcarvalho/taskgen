n_cpu_core=8
n_gpu=4

experiment_title='starter'
runs_per_setting=2 # number of seeds per settings
contexts_per_gpu=1 # number of runs to share on 1 GPU


filename_skip=['room_size', 'n_steps', 'log_interval_steps', 'replay_size', 'model', 'eval_max_trajectories']

search_space=[
  dict(
    sampler=dict(
            batch_B=8,
            eval_n_envs=2,
            ),
    runner=dict(
        n_steps=5e7, # 1e6=1 million, 1e8=100 million
        log_interval_steps=50e3, # 100K
    ),
    model=dict(
        out_conv=[0, 32],
        ),
    ),
]
