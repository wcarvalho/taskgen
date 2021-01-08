from gym import spaces


class GenericWrapper:
  def __init__(self, env):
    super(GenericWrapper, self).__init__()
    self.env = env

  def __getattr__(self, name):
    # if name.startswith('_'):
    #   raise AttributeError("attempted to get missing private attribute '{}'".format(name))
    return getattr(self.env, name)

# class EnvWrapper(object):
#     """docstring for EnvWrapper"""
#     def __init__(self, arg):
#         super(EnvWrapper, self).__init__()
#         self.arg = arg
        

class RGBImgPartialObsWrapper(GenericWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def step(self, *args, **kwargs):
        obs, action, reward, info = self.env.step(*args, **kwargs)

        return self.convert_observation(obs), action, reward, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)

        return self.convert_observation(obs)

    def convert_observation(self, obs):

        rgb_img_partial = self.env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial
        }
