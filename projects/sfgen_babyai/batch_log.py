"""
Run from root directory:
    MKL_THREADING_LAYER=GNU python experiments/set.py --log launchers/sfgen/batch_log

Run with breakpoint:
    MKL_THREADING_LAYER=GNU python -m ipdb -c continue experiments/set.py --log launchers/sfgen/batch_log
"""
import os.path
def shortener(key, value):
  if key == 'task_file':
    return os.path.basename(value)

  return value


n_cpu_core=16
n_gpu=4

runs_per_setting=3 # number of seeds per settings
contexts_per_gpu=2 # number of runs to share on 1 GPU

""" ======================================================
- search over: gvf

====================================================== """
experiment_title='benchmark'
filename_skip=[
  'room_size',
  'n_steps',
  'log_interval_steps',
  'replay_size',
  'eval_max_trajectories'
  ]


common_space=dict(
    level=dict(
        room_size=[8],
    ),
    env=dict(
        task_file=[
          "tasks/babyai_kitchen/simple_pickup.yaml",
          "tasks/babyai_kitchen/unseen_arg/length=2_slice_chill.yaml",
          "tasks/babyai_kitchen/unseen_arg/length=3_cook.yaml"
        ],
        ),
    runner=dict(
        # n_steps=[5e6], # 5 million
        n_steps=[50e6], # 50 million
        log_interval_steps=[50e6/100],
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
              model=['lstm_dqn', 'lstm_gvf']
          ),
        ),
]
