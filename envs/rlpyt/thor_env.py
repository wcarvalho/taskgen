"""
Class used to define rlpyt compatible environment for `babyai` and `babyai_kitchen`.
"""

import copy

import gym
import numpy as np
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import namedarraytuple, namedtuple
import ai2thor.controller
from torchvision import transforms
from PIL import Image


from envs.thor_nav.env import ThorNavEnv

EnvInfo = namedtuple("EnvInfo", [
    'success',
    ])  # Define in env file.

Observation = namedarraytuple(
    "Observation", 
    ["image", "task"])

def AlfredImageTransform():
  return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
class ThorEnv(Env):
    """
    The learning task, e.g. an MDP containing a transition function T(state,
    action)-->state'.  Has a defined observation space and action space.
    """

    def __init__(self,
        controller=ai2thor.controller.Controller,
        env_class=ThorNavEnv,
        reward_scale=1,
        timestep_penalty=-0.04,
        seed=1,
        verbosity=0,
        **kwargs,
        ):

        super(ThorEnv, self).__init__()
        # ======================================================
        # load environment
        # ======================================================
        self.timestep_penalty = timestep_penalty
        self.reward_scale = reward_scale
        self.verbosity = verbosity

        self.env = env_class(controller=controller, **kwargs)
        self._seed = seed
        self.transform = AlfredImageTransform()

    def seed(self, seed):
        self.env.set_seed(seed)
        self._seed = seed

    def process_obs(self, obs):
        # -----------------------
        # channel 1st for image
        # -----------------------
        image = Image.fromarray(obs['image'])
        image = self.transform(image)
        obs['image'] = image

        return Observation(**obs)

    def step(self, action):
        """
        """
        obs, reward, done, info = self.env.step(action)

        if not 'success' in info:
            info['success'] = reward > 0
        info = EnvInfo(**info)
        obs = self.process_obs(obs)

        reward = self.update_reward(reward)
        return EnvStep(obs, reward, done, info)

    def update_reward(self, reward):
        reward = reward + self.timestep_penalty
        reward = reward*self.reward_scale
        return reward

    def reset(self):
        obs = self.env.reset()
        obs = self.process_obs(obs)
        return obs

    @property
    def action_space(self):
        return GymSpaceWrapper(self.env.action_space)

    @property
    def observation_space(self):
        return GymSpaceWrapper(self.env.observation_space)

    # @property
    # def horizon(self):
    #     """Episode horizon of the environment, if it has one."""
    #     return self.env.max_steps
