
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