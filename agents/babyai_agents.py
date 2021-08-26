from rlpyt.utils.buffer import buffer_to


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


    def get_variables(self, observation, prev_action, prev_reward, init_rnn_state, target=False, **kwargs):
        # Assume init_rnn_state already shaped: [N,B,H]
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)

        if target:
            return self.target_model(*model_inputs, **kwargs)
        else:
            return self.model(*model_inputs, **kwargs)




# ======================================================
# PPO
# ======================================================
# copied from: rlpyt.agents.pg.atari:AtariLstmAgent
from rlpyt.agents.pg.categorical import RecurrentCategoricalPgAgent
# from sfgen.babyai.preloads import BabyAIPPOModel
class BabyAIPPOAgent(BabyAIMixin, RecurrentCategoricalPgAgent):
    def __init__(self, ModelCls=BabyAIRLModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


    def get_variables(self, observation, prev_action, prev_reward, init_rnn_state, **kwargs):
        # Assume init_rnn_state already shaped: [N,B,H]
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)
        return self.model(*model_inputs, **kwargs)
