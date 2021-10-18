n_cpu_core=8
n_gpu=4

experiment_title='starter'
runs_per_setting=2 # number of seeds per settings
contexts_per_gpu=1 # number of runs to share on 1 GPU


filename_skip=['n_steps', 'log_interval_steps', 'model']

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
        out_conv=[32],
        ),
    algo=dict(
        entropy_loss_coeff=[0.01, 0.005],
        ),
    ),
]
