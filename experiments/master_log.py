
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
# 2021.01.19 - RLDL
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
2021.01.23/4 - RLDL: _rlpyt/data/local/{2021.01.23, 2021.01.24}
    - had to rerun because 5e7 wasn't long enough
how do the following dimensions effect performance:
- whether to decay learning rate (IMPORTANT)
- size of view (unclear results)
- weight decay (FiLM said important by babyai doesn't use?)
RERUN with more env steps (100 million) + smaller room
answers:
- 12
- lower learning rate did better
- it looks like smaller view does better? got to about 80% success
====================================================== """
experiment_title='lrdecay_view_weightdecay2'
runs_per_setting=2
search_space={
    'settings' : {
        'config' : ['ppo_babyai']
    },
    'algo' : {
        'learning_rate' : [5e-5],
        'linear_lr_schedule' : [True, False],
    },
    'optim': {
        'weight_decay' : [0, 1e-5],
    },
    'env': {
        'level' : ["PutNextLocal"],
    },
    'level' : {
        'agent_view_size' : [7, 3],
    },
    'runner' : dict(
        n_steps=[1e8], # 50 million
    )
}