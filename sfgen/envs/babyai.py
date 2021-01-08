import gym
import babyai

from rlpyt.envs.base import Env, EnvStep

# class BabyAIEnv(object):
#     """docstring for BabyAIEnv"""
#     def __init__(self, arg):
#         super(BabyAIEnv, self).__init__()
#         self.arg = arg

class BabyAIEnv(Env):
    """
    The learning task, e.g. an MDP containing a transition function T(state,
    action)-->state'.  Has a defined observation space and action space.
    """
    def __init__(self, level):
        super(BabyAIEnv, self).__init__()
        self.level = level
        self.env = gym.make(level)

        import ipdb; ipdb.set_trace()

    def step(self, action):
        """
        Run on timestep of the environment's dynamics using the input action,
        advancing the internal state; T(state,action)-->state'.

        Args:
            action: An element of this environment's action space.
        
        Returns:
            observation: An element of this environment's observation space corresponding to the next state.
            reward (float): A scalar reward resulting from the state transition.
            done (bool): Indicates whether the episode has ended.
            info (namedtuple): Additional custom information.
        """

        import ipdb; ipdb.set_trace()

    def reset(self):
        """
        Resets the state of the environment.

        Returns:
            observation: The initial observation of the new episode.
        """
        import ipdb; ipdb.set_trace()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    @property
    def horizon(self):
        """Episode horizon of the environment, if it has one."""
        import ipdb; ipdb.set_trace()

    def close(self):
        """Any clean up operation."""
        pass
