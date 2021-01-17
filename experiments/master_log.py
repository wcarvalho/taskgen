
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
        'film_batch_norm' : [True, False],
        'film_pool' : [True, False],
        'intrustion_policy_input' : [True, False]
    },
    'runner' : dict(
        n_steps=[7.5e6],
    )
}