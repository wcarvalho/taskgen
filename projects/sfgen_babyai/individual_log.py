# PYTORCH_JIT=0 python -m ipdb -c continue experiments/individual.py --cuda_idx 2 --n_parallel 32 --verbosity 0
# python -m ipdb -c continue experiments/individual.py --cuda_idx 2 --n_parallel 32 --verbosity 0



""" ======================================================
Server, Testing
====================================================== """
config=dict(
    settings=dict(
        model='lstm_gvf',
    ),
    env=dict(
        tasks_file="tasks/babyai_kitchen/unseen_arg/length=2_slice_chill.yaml",
        ),
    level=dict(
        room_size=8,
    ),
    gvf=dict(
        coeff=.01,
        stop_grad=True,
        ),
    runner=dict(
        n_steps=5e7, # 1e6=1 million, 1e8=100 million
        log_interval_steps=1e4,
    ),
    algo=dict(
        min_steps_learn=int(5e3),
        store_rnn_state_interval=40,
        ),
    sampler=dict(
        batch_T=40,
        batch_B=2,
        eval_n_envs=2,
        eval_max_trajectories=10,
        )
)