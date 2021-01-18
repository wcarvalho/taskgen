import copy

# import babyai.rl
# import babyai.levels.levelgen as levelgen
import babyai.levels.iclr19_levels as iclr19_levels
import gym
import numpy as np
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import namedarraytuple, namedtuple

EnvInfo = namedtuple("EnvInfo", [])  # Define in env file.

PixelObservation = namedarraytuple("PixelObservation", ["image", "mission"])
SymbolicObservation = namedarraytuple("SymbolicObservation", ["image", "mission", "direction"])


class BabyAIEnv(Env):
    """
    The learning task, e.g. an MDP containing a transition function T(state,
    action)-->state'.  Has a defined observation space and action space.
    """
    def __init__(self,
        level,
        reward_scale=1,
        instr_preprocessor=None,
        max_sentence_length=20,
        use_pixels=True,
        num_missions=0,
        seed=0,
        verbosity=0,
        env_kwargs={},
        ):
        super(BabyAIEnv, self).__init__()
        # ======================================================
        # dynamically load relevant "level" env
        # ======================================================
        self.level = level
        self.reward_scale = reward_scale
        self.verbosity = verbosity
        self.env_class = getattr(iclr19_levels, f"Level_{level}")
        self.env = self.env_class(**env_kwargs)
        self._seed = 1

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
        if use_pixels:
            self.env = RGBImgPartialObsWrapper(self.env)


        # -----------------------
        # class to store obs info
        # -----------------------
        # self._obs = namedarraytuple("Observation", [k for k in self.observation_space._gym_space.spaces.keys()])

    def seed(self, seed):
        self.env.seed(seed)
        self._seed = seed

    def process_obs(self, obs):
        # -----------------------
        # channel 1st for image
        # -----------------------
        obs['image'] = obs['image'].transpose(2,0,1)

        # -----------------------
        # get tokens
        # -----------------------
        if self.instr_preprocessor:
            obs['mission'] = self.instr_preprocessor([obs], torchify=False)[0]

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
        # if self.verbosity:
            # print(f"Mission: {obs['mission']}")
        obs = self.process_obs(obs)
        info = EnvInfo(**info)

        reward = reward*self.reward_scale
        return EnvStep(obs, reward, done, info)

    def reset(self):
        """
        """
        if self.num_missions:
            seed = np.random.randint(self.num_missions)
            if self.verbosity:
                print(f"Samplled mission {seed}/{self.num_missions}")
            seed = 1000*(self._seed) + seed
            self.env.seed(seed)

        obs = self.env.reset()

        if self.verbosity:
            print(f"Mission: {obs['mission']}")
        # import ipdb; ipdb.set_trace()
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
                num_possible_tokens = len(self.instr_preprocessor.vocab.vocab)
                assert num_possible_tokens > 0, "vocabulary is empty"
                if num_possible_tokens <= 255:
                    mission_dtype=np.uint8
                else:
                    mission_dtype=np.int32

                observation_space.spaces['mission'] = gym.spaces.Box(
                  low=0, high=num_possible_tokens, shape=(self.max_sentence_length,num_possible_tokens), dtype=mission_dtype
                )


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
        print("when is this used?")
        import ipdb; ipdb.set_trace()
        return self.env.max_steps



class MultiLevelBabyAIEnv(BabyAIEnv):
    """docstring for MultiLevelBabyAIEnv"""
    def __init__(self, levels, env_kwargs, level_kwargs):
        self.envs = {}
        for level in levels:
            kwargs = copy.deepcopy(env_kwargs)
            if level in level_kwargs:
                kwargs.update(level_kwargs[level])
            env = BabyAIEnv(level, **kwargs)
            self.envs[level] = env

        self.init_env = env
        import ipdb; ipdb.set_trace()


    def step(self, action):
        """
        """
        obs_dict, action, reward, info = self.env.step(action)
        import ipdb; ipdb.set_trace()
        return EnvStep(obs_dict['image'], action, reward, info)

    def reset(self):
        """
        """
        if self.num_missions:
            seed = np.random.randint(self.num_missions)
            return self.env.reset(seed)
        else:
            return self.env.reset()

    @property
    def action_space(self):
        return self.init_env.action_space

    @property
    def observation_space(self):
        return self.init_env.observation_space

    @property
    def horizon(self):
        """Episode horizon of the environment, if it has one."""
        print("when is this used?")
        import ipdb; ipdb.set_trace()
        return self.env.max_steps



# # ======================================================
# # Helpers
# # ======================================================
# import numpy as np
# import string
# import random
# import gym


# class Token(gym.Space):
#     def __init__(
#                 self,
#                 length=None,
#                 min_length=1,
#                 max_length=180,
#             ):
#         self.length = length
#         self.min_length = min_length
#         self.max_length = max_length
#         self.letters = string.ascii_letters + " .,!-"

#     def sample(self):
#         length = random.randint(self.min_length, self.max_length)
#         string = ""
#         for i in range(length):
#             letter = random.choice(self.letters)
#             string += letter
#         return string

#     def contains(self, x):
#         is_a_string = isinstance(x, str)
#         correct_length = self.min_length < len(x) < self.max_length
#         correct_letters = all([l in self.letters for l in x])
#         return is_a_string and correct_length and correct_letters