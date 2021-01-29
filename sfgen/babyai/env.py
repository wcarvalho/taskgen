import copy

import gym
import numpy as np
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import namedarraytuple, namedtuple
from sfgen.babyai_kitchen.levelgen import KitchenLevel

EnvInfo = namedtuple("EnvInfo", ['success'])  # Define in env file.

PixelObservation = namedarraytuple("PixelObservation", ["image", "mission", "mission_idx"])
SymbolicObservation = namedarraytuple("SymbolicObservation", ["image", "mission", "mission_idx", "direction"])


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
        strict_task_idx_loading=True,
        **kwargs,
        ):
        super(BabyAIEnv, self).__init__()
        # ======================================================
        # load environment
        # ======================================================
        self.reward_scale = reward_scale
        self.verbosity = verbosity

        if 'num_grid' in level_kwargs:
            ncells = level_kwargs.pop('num_grid')
            level_kwargs['num_rows'] = ncells
            level_kwargs['num_cols'] = ncells

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
        if use_pixels:
            self.env = RGBImgPartialObsWrapper(self.env)


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
        if obs['mission'] in self.task2idx:
            idx = self.task2idx[obs['mission']]
            obs['mission_idx'] = idx
        else:
            if strict_task_idx_loading:
                raise RuntimeError(f"Encountered unknown task: {obs['mission']}")
        # -----------------------
        # get tokens
        # -----------------------
        if self.instr_preprocessor:
            mission = np.zeros(self.max_sentence_length)
            indices = self.instr_preprocessor([obs], torchify=False)[0]
            assert len(indices) < self.max_sentence_length, "need to increase sentence length capacity"
            mission[:len(indices)] = indices
            obs['mission'] = mission


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

        obs = self.process_obs(obs)
        if not 'success' in info:
            import ipdb; ipdb.set_trace()
            if reward > 0: 
                info['success'] = True

        info = EnvInfo(**info)

        reward = reward*self.reward_scale
        return EnvStep(obs, reward, done, info)

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
