# PYTORCH_JIT=0 python -m ipdb -c continue experiments/individual.py --cuda_idx 2 --n_parallel 32 --verbosity 0
# python -m ipdb -c continue experiments/individual.py --cuda_idx 2 --n_parallel 32 --verbosity 0



""" ======================================================
2021.02.10 - Brain
    - testing object-model
====================================================== """
config=dict(
    settings=dict(
        # collector='reg',
        # aux='cont_obj_model',
        gvf='goal_gvf',
    ),
    env=dict(
        task_file="cool_slice_place_heat_01.yaml",
        ),
    level=dict(
        num_dists=9,
        room_size=7,
    ),
    gvf=dict(
        coeff=.01,
        stop_grad=True,
        ),
    model=dict(
        nheads=4,
        default_size=512,
        individual_rnn_dim=128,
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
        )
)