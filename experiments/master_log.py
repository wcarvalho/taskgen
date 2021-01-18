
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
# 2021.01.17 - RLDL
# got language to work. 
# - batchnorm was critical.
# - pool with film helped a lot as well
# - giving instruction to policy seemed to help a little
# runs: _rlpyt/data/local/20210117/152041/lang_focus
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
# is language as input to policy important?
# ======================================================
experiment_title='lang_policy'
runs_per_setting=2
search_space={
    'algorithm' : {'algorithm' : ['ppo_babyai']},
    'env': {
        'level' : ["GoTo"],
        'level' : ["GoToSeq"],
    },
    # 'model' : {
    #     'intrustion_policy_input' : [True]
    # },
    'runner' : dict(
        n_steps=[1e7],
    ),
    'level' : dict(
        num_grid=[3, 2],
        )
}