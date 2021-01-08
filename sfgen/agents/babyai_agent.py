from rlpyt.agents.dqn.r2d1_agent import R2d1Agent, R2d1AlternatingAgent
# from rlpyt.agents.dqn.atari.mixin import AtariMixin


# from rlpyt.models.dqn.atari_r2d1_model import AtariR2d1Model
from sfgen.models.babyai_r2d1_model import BabyAIR2d1Model

class BabyAIMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(output_size=env_spaces.action.n,
            **{f"{k}_shape":v.shape for k,v in env_spaces.observation._gym_space.spaces.items()})


class BabyAIR2d1Agent(BabyAIMixin, R2d1Agent):

    def __init__(self, ModelCls=BabyAIR2d1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
