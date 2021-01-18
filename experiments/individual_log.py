# ======================================================
# 2021.01.17 - RLDL
# replicating babyAI architecture
# ======================================================
config=dict(
    model=dict(
        batch_norm=True,
        film_batch_norm=True,
        film_pool=True,
        intrustion_policy_input=False,
        ),
    env=dict(
        level='GoToSeq',
        ),
    level=dict(
        num_grid=2,
        )
)
