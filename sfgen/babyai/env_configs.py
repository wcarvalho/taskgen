import copy

configs = dict()

# ======================================================
# BabyAI env
# ======================================================
config = dict(
    settings=dict(
        env='babyai',
    ),
    env=dict(
        level="GoToLocal",
        use_pixels=True,
        num_missions=0,
    )
)
configs["babyai"] = config

# ======================================================
# Kitchen env
# ======================================================
config = copy.deepcopy(configs["babyai"])
config.update(dict(
    settings=dict(
        env='babyai_kitchen',
    ),
    env=dict(
        task_kinds=['slice', 'cool'],
        actions = ['left', 'right', 'forward', 'pickup_container', 'pickup_contents', 'place', 'toggle', 'slice'],
        room_size=8,
        agent_view_size=7,
        num_dists=5,
        random_object_state=False,
        use_time_limit=True,
        )))
configs["babyai_kitchen"] = config
