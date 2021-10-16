experiment_title='starter'
runs_per_setting=2 # number of seeds per settings
contexts_per_gpu=2 # number of runs to share on 1 GPU
search_space=[
    dict(
        env=dict(
            level=["GoToLocal", "PutNextLocal"],
            ),
        ),
]
