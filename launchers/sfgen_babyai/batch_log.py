"""
Run from root directory:
    MKL_THREADING_LAYER=GNU python experiments/set.py --log launchers/sfgen/batch_log

Run with breakpoint:
    MKL_THREADING_LAYER=GNU python -m ipdb -c continue experiments/set.py --log launchers/sfgen/batch_log
"""

""" ======================================================
2021.02.23 - RLDL
- search over: gvf

====================================================== """
experiment_title='gvf_3'
runs_per_setting=2
contexts_per_gpu=2
filename_skip=['room_size', 'n_steps', 'log_interval_steps', 'replay_size', 'model', 'eval_max_trajectories']
common_space=dict(
    level=dict(
        # num_dists=[0, 2],
        # room_size=[5],
        num_dists=[3, 6],
        room_size=[6],
    ),
    env=dict(
        # task_file=["cool_slice_place_heat_01.yaml"],
        task_file=["test_cool_slice_01.yaml"],
        ),
    runner=dict(
        # n_steps=[5e6], # 5 million
        n_steps=[1e7], # 5 million
        log_interval_steps=[20e4],
    ),
    algo=dict(
        # eps_steps=[5e6], # 10 million
        # eps_steps=[1e7], # 10 million
        replay_size=[int(5e5)],
    ),
    sampler=dict(
        eval_max_trajectories=[500],
        ),
)
""" -----------
SFGEN
----------- """
search_space=[
    # impact of number of RNN heads
    # size = 8
    dict(
        **common_space,
        settings=dict(
            gvf=['goal_gvf'],
        ),
        gvf=dict(
            coeff=[1e-3, 1e-4, 1e-5, 0],
            stop_grad=[True],
            ),
        model=dict(
            nheads=[1, 8],
            individual_rnn_dim=[128],
            default_size=[1024],
            ),
        ),
]
