# MKL_THREADING_LAYER=GNU python -m ipdb -c continue experiments/babyai_exp_set_master.py

# ======================================================
# 2021.01.16 - RLDL
# ======================================================
experiment_title='ppo_dqn'
runs_per_setting=2
search_space={
    'algorithm' : {'algorithm' : ['ppo_babyai']},
    'env': {
        'level' : ["GoToRedBall", "GoToLocal"]
    },
}

# ======================================================
# 2021.01.17 - RLDL
# getting language to work
# ======================================================
experiment_title='lang_focus'
runs_per_setting=1
search_space={
    'algorithm' : {'algorithm' : ['ppo_babyai']},
    'env': {
        'level' : ["GoToLocal"]
    },
    'model' : {
        'batch_norm' : [True, False],
        # 'film_batch_norm' : [False],
        'film_pool' : [True, False],
        'intrustion_policy_input' : [True, False]
    },
    'runner' : dict(
        n_steps=[7.5e6],
    )
}

# ======================================================
# 2021.01.17 - RLDL: _rlpyt/data/local/20210117/152041/lang_focus
# got language to work. 
# - batchnorm was critical.
# - pool with film helped a lot as well
# - giving instruction to policy seemed to help a little
# blow: is it critical because of FILM or CNN?
# - answer: CNN. want batchnorm overall
# ======================================================
experiment_title='lang_focus'
runs_per_setting=2
search_space={
    'algorithm' : {'algorithm' : ['ppo_babyai']},
    'env': {
        'level' : ["GoToLocal", "PickupLoc"]
    },
    'model' : {
        'batch_norm' : [False, True],
        'film_batch_norm' : [True],
        'film_pool' : [True],
        'intrustion_policy_input' : [True]
    },
    'runner' : dict(
        n_steps=[7.5e6],
    )
}


# ======================================================
# 2021.01.18 - RLDL
# how does policy do across "hard" tasks?
# all failed except: "open"
# ======================================================
experiment_title='lang_policy_all'
runs_per_setting=2
search_space={
    'algorithm' : {'algorithm' : ['ppo_babyai']},
    'env': {
        'level' : [
            "GoTo", #<--- in paper
            "Pickup",
            "UnblockPickup",
            "Open",
            "Unlock",
            "PutNext",
            "Synth",
            "GoToSeq",
        ],
    },
    'model' : {
        'intrustion_policy_input' : [True]
    },
    'sampler' : {
        'eval_max_steps' : [int(100e3)],
    },
    'runner' : dict(
        n_steps=[1e7],
    ),
    'level' : dict(
        num_grid=[3],
        )
}


# ======================================================
# 2021.01.19 - RLDL: (deleted dir)
# how do following dimensions effect performance:
# - partial observability view size
# - size of room? 
# ANSWER: beyond 8 seems too hard... 
# for some reason, smaller window leads to better performance??
# ======================================================
experiment_title='dims_of_difficulty'
runs_per_setting=2
search_space={
    'algo' : {
        'learning_rate' : [1e-4, 5e-5],
    },
    'algorithm' : {'algorithm' : ['ppo_babyai']},
    'env': {
        'level' : ["PutNextLocal"],
    },
    'level' : {
        'agent_view_size' : [7, 5, 3],
        'room_size' : [8, 12, 16],
    },
    'runner' : dict(
        n_steps=[2.5e7],
    )
}



""" ======================================================
2021.01.20 - RLDL: _rlpyt/data/local/20210120/
how do following dimensions effect performance:
- partial observability view size
- size of room? 
RERUN with more env steps (100 million) + smaller room
answers:
- 12
    - view=7 failed
    - view = {3,5} did about same. high variance: 30-80% on 2 seeds
- 8
    - lower learning rate did better
    - view=3 learned fastest, view=5 afterwards, view=7 last.
- it looks like lower learning rate does do better? got to about 80% success twice
====================================================== """
experiment_title='dims_of_difficulty'
runs_per_setting=2
search_space={
    'algo' : {
        'learning_rate' : [1e-4, 5e-5],
    },
    'algorithm' : {'algorithm' : ['ppo_babyai']},
    'env': {
        'level' : ["PutNextLocal"],
    },
    'level' : {
        'agent_view_size' : [7, 5, 3],
        'room_size' : [8, 12],
    },
    'runner' : dict(
        n_steps=[1e8],
    )
}


""" ======================================================
2021.01.23/4/7 - RLDL: _rlpyt/data/local/{2021.01.23, 2021.01.24, 2021.01.27}
    - had to rerun because 5e7 wasn't long enough
    - had to rerun again because ethan accidentally killed jobs
how do the following dimensions effect performance:
- whether to decay learning rate (IMPORTANT)
- size of view (unclear results)
- weight decay (FiLM said important by babyai doesn't use?)
====================================================== """
experiment_title='lrdecay_view_weightdecay3'
runs_per_setting=2
search_space={
    'settings' : {
        'config' : ['ppo_babyai']
    },
    'algo' : {
        'learning_rate' : [5e-5],
        'linear_lr_schedule' : [
            # True,
            False
        ],
    },
    'optim': {
        'weight_decay' : [
            0,
            1e-5
        ],
    },
    'env': {
        'level' : ["PutNextLocal"],
    },
    'level' : {
        'agent_view_size' : [
            7,
            # 3
        ],
    },
    'runner' : dict(
        n_steps=[1.5e8], # 50 million
    )
}


""" ======================================================
2021.01.30 - Brain: 
- seeing how the baselines do on basic versions of the tasks I made
- pick the better performing and use it as the baseline of comparison from here only
====================================================== """
experiment_title='kitchen_baselines'
runs_per_setting=2
n_cpu_core=16
n_gpu=4
contexts_per_gpu=1
search_space=dict(
    settings=dict(
        agent=['babyai_ppo', 'chaplot_ppo'],
        # agent=['chaplot_ppo'],
        env=['babyai_kitchen'],
    ),
    level=dict(
        task_kinds=[
            ['heat'],
            ['slice'],
            ['slice', 'cool'],
        ],
        num_dists=[
            # 0,
            # 5,
            10
        ],
    )

)




""" ======================================================
2021.01.30 - Brain: 
- seeing how the baselines do on basic versions of the tasks I made
- pick the better performing and use it as the baseline of comparison from here only
====================================================== """
experiment_title='kitchen_baselines_2'
runs_per_setting=2
n_cpu_core=16
n_gpu=4
contexts_per_gpu=2
search_space=dict(
    settings=dict(
        agent=['chaplot_ppo'],
        env=['babyai_kitchen'],
    ),
    level=dict(
        task_kinds=[
            # ['place'],
            # ['cool'],
            ['cook'],
        ],
        num_dists=[
            0,
            5,
            10
        ],
    ),
    runner=dict(
        n_steps=[5e7], # 50 million
    )

)



""" ======================================================
2021.02.{01,02,03} - Brain
- seeing how dqn + chaplot does.
====================================================== """
experiment_title='kitchen_baselines_dqn_4'
runs_per_setting=1
n_cpu_core=32
n_gpu=3
contexts_per_gpu=2
search_space=dict(
    settings=dict(
        model=['chaplot'],
        algorithm=['r2d1', 'ppo'],
        env=['babyai_kitchen'],
    ),
    level=dict(
        task_kinds=[
            ['slice'],
            ['heat'],
            # ['cool'],
        ],
        num_dists=[0,5],
    ),
    agent=dict(
        eps_eval=[0.1, 0.01],
        ),
    # model=dict(
    #     batch_norm=[False, True],
    # ),
    algo=dict(
        eps_steps=[5e6], # 5 million
        warmup_T=[0, 20],
        replay_ratio=[1, 4]
    ),
    runner=dict(
        n_steps=[10e6], # 5 million
        log_interval_steps=[10e4],
    ),
    env=dict(
        timestep_penalty=[-0.004],
        )

)


# """ ======================================================
# 2021.02.{01-02} - Brain
# - seeing how dqn + babyai does.

# ====================================================== """
experiment_title='kitchen_baselines_dqn_4'
runs_per_setting=1
n_cpu_core=32
n_gpu=2
contexts_per_gpu=2
search_space=dict(
    settings=dict(
        model=['babyai'],
        algorithm=['r2d1', 'ppo'],
        env=['babyai_kitchen'],
    ),
    level=dict(
        task_kinds=[
            ['slice'],
            ['heat'],
        ],
        num_dists=[0, 5],
    ),
    agent=dict(
        eps_eval=[0.1, 0.01],
        ),
    # model=dict(
    #     batch_norm = [False],
    #     film_bias  = [False],
    #     film_batch_norm = [False],
    # ),
    algo=dict(
        eps_steps=[5e6], # 5 million
        warmup_T=[0, 20],
        replay_ratio=[1, 4]
    ),
    runner=dict(
        n_steps=[10e6], # 5 million
        log_interval_steps=[10e4],
    ),
    env=dict(
        timestep_penalty=[ -0.004],
        )
)