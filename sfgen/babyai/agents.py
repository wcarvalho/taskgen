from sfgen.babyai.babyai_model import BabyAIRLModel


# copied from: rlpyt.agents.dqn.atari.mixin:AtariMixin
class BabyAIMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(output_size=env_spaces.action.n,
            **{f"{k}_shape":v.shape for k,v in env_spaces.observation._gym_space.spaces.items()})

# ======================================================
# DQN
# ======================================================
# copied from: rlpyt.agents.dqn.atari.atari_r2d1_agent:AtariR2d1Agent
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
class BabyAIR2d1Agent(BabyAIMixin, R2d1Agent):

    def __init__(self, ModelCls=BabyAIRLModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)





# ======================================================
# PPO
# ======================================================
# copied from: rlpyt.agents.pg.atari:AtariLstmAgent
from rlpyt.agents.pg.categorical import RecurrentCategoricalPgAgent
# from sfgen.babyai.models import BabyAIPPOModel
class BabyAIPPOAgent(BabyAIMixin, RecurrentCategoricalPgAgent):
    def __init__(self, ModelCls=BabyAIRLModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


