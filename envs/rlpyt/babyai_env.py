"""
Class used to define rlpyt compatible environment for `babyai` and `babyai_kitchen`.
"""

import copy

import gym
import numpy as np
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import namedarraytuple, namedtuple
from envs.babyai_kitchen.levelgen import KitchenLevel

from rlpyt.utils.collections import AttrDict

EnvInfo = namedtuple("EnvInfo", [
    'success',
    # 'task',
    # 'level',
    ])  # Define in env file.

PixelObservation = namedarraytuple(
    "PixelObservation", 
    ["image", "mission", "mission_idx"])
SymbolicObservation = namedarraytuple(
    "SymbolicObservation",
    ["image", "mission", "mission_idx", "direction"])


class BabyAIEnv(Env):
    """
    The learning task, e.g. an MDP containing a transition function T(state,
    action)-->state'.  Has a defined observation space and action space.
    """

    def __init__(self,
        env_class=KitchenLevel,
        reward_scale=1,
        instr_preprocessor=None,
        max_sentence_length=50,
        use_pixels=True,
        num_missions=0,
        seed=1,
        verbosity=0,
        level_kwargs={},
        task2idx={},
        valid_tasks=None,
        tile_size=36,
        timestep_penalty=-0.04,
        strict_task_idx_loading=False,
        **kwargs,
        ):

        super(BabyAIEnv, self).__init__()
        # ======================================================
        # load environment
        # ======================================================
        self.reward_scale = reward_scale
        self.verbosity = verbosity
        self.timestep_penalty = timestep_penalty

        if 'num_grid' in level_kwargs:
            ncells = level_kwargs.pop('num_grid')
            level_kwargs['num_rows'] = ncells
            level_kwargs['num_cols'] = ncells

        # if not 'verbosity' in level_kwargs:
        #     level_kwargs['verbosity'] = verbosity

        if valid_tasks is not None:
            level_kwargs['valid_tasks'] = valid_tasks

        self.env_class = env_class
        self.env = env_class(**level_kwargs, seed=seed)
        self._seed = seed

        # -----------------------
        # stuff for loading task indices
        # -----------------------
        self.task2idx = task2idx
        self.strict_task_idx_loading = strict_task_idx_loading
        # -----------------------
        # stuff to load language
        # -----------------------
        self.num_missions = num_missions
        self.instr_preprocessor = instr_preprocessor
        self.max_sentence_length = max_sentence_length

        # -----------------------
        # pixel observation
        # -----------------------
        self.use_pixels = use_pixels
        self.tile_size = tile_size
        if use_pixels:
          self.env = RGBImgPartialObsWrapper(self.env, tile_size=tile_size)


    def seed(self, seed):
        self.env.seed(seed)
        self._seed = seed

    def process_obs(self, obs):
        # -----------------------
        # channel 1st for image
        # -----------------------
        obs['image'] = obs['image'].transpose(2,0,1)

        # -----------------------
        # get task index
        # -----------------------
        obs['mission_idx'] = obs.get('mission_idx', 0)

        # -----------------------
        # get tokens
        # -----------------------
        if self.instr_preprocessor:
          obs['mission'] = self.instr_preprocessor(obs['mission'])

        # -----------------------
        # get direction
        # -----------------------
        if not self.use_pixels:
            raise NotImplemented

        if self.use_pixels:
            return PixelObservation(**obs)
        else:
            return SymbolicObservation(**obs)

    def step(self, action):
        """
        """
        obs, reward, done, info = self.env.step(action)

        if not 'success' in info:
            info['success'] = reward > 0
        # info['task'] = obs['mission']
        # info['level'] = obs.get('level', 'none')
        info = EnvInfo(**info)

        obs = self.process_obs(obs)
        reward = self.update_reward(reward)
        return EnvStep(obs, reward, done, info)

    def update_reward(self, reward):
        reward = reward + self.timestep_penalty
        reward = reward*self.reward_scale
        return reward

    def reset(self):
        """
        """
        if self.verbosity:
            print("-"*50)
        if self.num_missions:
            seed = np.random.randint(self.num_missions)
            if self.verbosity:
                print(f"Sampled mission {seed}/{self.num_missions}")
            seed = 1000*(self._seed) + seed
            self.env.seed(seed)

        obs = self.env.reset()

        if self.verbosity:
            print(f"Mission: {obs['mission']}. Timelimit: {self.env.max_steps}")
        obs = self.process_obs(obs)
        return obs

    @property
    def action_space(self):
        return GymSpaceWrapper(self.env.action_space)

    @property
    def observation_space(self):
        env = self.env
        observation_space = copy.deepcopy(env.observation_space)
        # -----------------------
        # image
        # -----------------------
        im_shape = observation_space.spaces['image'].shape
        observation_space.spaces['image'] = gym.spaces.Box(
              low=0, high=255, shape=(im_shape[2], im_shape[0], im_shape[1]), dtype=np.uint8
            )

        # -----------------------
        # missions
        # -----------------------
        if self.instr_preprocessor:
            if self.num_missions == 1:
                pass
            else:
                observation_space.spaces['mission'] = self.instr_preprocessor.gym_observation_space()


        # -----------------------
        # direction
        # -----------------------
        if self.use_pixels:
           # don't need to do anything
           pass
        else:
            # direction can't be inferred from 
            observation_space.spaces['direction'] = gym.spaces.Box(
              low=0, high=4, shape=(1,), dtype=np.uint8
            )

        return GymSpaceWrapper(observation_space)

    @property
    def horizon(self):
        """Episode horizon of the environment, if it has one."""
        return self.env.max_steps



class BabyAITrajInfo(AttrDict):
    """
    Tr
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.success = False
        self.DiscountedReturn = 0
        self._cur_discount = 1
        self._task = None

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        self.success = self.success or env_info.success
        self.DiscountedReturn += self._cur_discount * reward
        self._task = observation.mission_idx
        self._cur_discount *= self._discount

    def terminate(self, observation):
        return self
