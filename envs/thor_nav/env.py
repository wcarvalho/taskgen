import gym
from gym import spaces
import numpy as np
import ai2thor.controller
from envs.thor_nav import ALFRED_CONSTANTS
import random

class ThorNavEnv(gym.Env):
  """docstring for ThorObjectNav"""
  def __init__(self,
    actions=None,
    floorplans=None,
    success_distance=2.0,
    task_dist=1.0, # how far away (after success) should initial task objects be
    task_dist_step=1.0, # how much should distance increase after success
    tasks_per_floorplan_reset=50,
    controller_kwargs=None,
    init_kwargs=None,
    controller=ai2thor.controller.Controller,
    seed=1,
    max_steps=200,
    **kwargs):
    super(ThorNavEnv, self).__init__()

    assert floorplans is not None, "please set floorplans"

    self.seed = seed
    self.controller_kwargs = controller_kwargs or dict()
    self.init_kwargs = init_kwargs or dict()
    self.floorplans = floorplans
    self.floorplan = np.random.choice(self.floorplans)
    self.actions = actions or [
      "MoveAhead", "MoveBack", "RotateRight", "RotateLeft", "LookUp", "LookDown"
    ]
    # ======================================================
    # initialize controller
    # ======================================================
    self.controller = controller(**self.controller_kwargs)
    self.controller.start()
    event = self.controller.step(dict(
      action='Initialize',
      **self.init_kwargs)
      )
    self.reset_floorplan()

    # ======================================================
    # observation + action space
    # ======================================================
    width = init_kwargs.get('width', 300)
    height = init_kwargs.get('height', 300)
    self.tasks = set(ALFRED_CONSTANTS.OBJECTS)
    self.ntasks = len(self.tasks)
    self.observation_space = spaces.Dict({
      'image': spaces.Box(
        low=0,
        high=255,
        shape=(width, height, 3),
        dtype='uint8'),
      'task' : spaces.MultiBinary(self.ntasks)
    })
    self.object2idx = {o:idx for idx, o in enumerate(ALFRED_CONSTANTS.OBJECTS)}
    self.action_space = spaces.Discrete(len(self.actions))

    # ======================================================
    # initialize task info
    # ======================================================
    self.tasks_in_floorplan = 0
    self.tasks_per_floorplan_reset = tasks_per_floorplan_reset
    self.success_distance = self.min_task_dist = success_distance
    self.task_dist = task_dist
    self.task_dist_step = task_dist_step
    self.max_steps = max_steps

  def set_seed(self, seed):
    self.seed = seed
    np.random.seed(seed)
    random.seed(seed)

  def randomize_object_locations(self):
    event = self.controller.step(dict(
      action='InitialRandomSpawn',
      randomSeed=self.seed,
      forceVisible=False,
    ))

    return event

  def randomize_agent(self):
    noptions = len(self.reachable_positions)
    coord = np.random.randint(noptions)
    rand_coord = self.reachable_positions[coord]

    horizon_optons = np.arange(-30, 60.1, self.init_kwargs.get('rotateHorizonDegrees', 30))
    rotation_optons = np.arange(0, 360.1, self.init_kwargs.get('rotateStepDegrees', 90))
    horizon = np.random.choice(horizon_optons)
    rotation = np.random.choice(rotation_optons)

    event = self.controller.step(dict(action='TeleportFull', **rand_coord, rotation=dict(y=rotation), horizon=horizon))

    return event

  def reset_floorplan(self):
    self.floorplan = np.random.choice(self.floorplans)
    self.tasks_in_floorplan = 0
    event = self.controller.reset(f"FloorPlan{self.floorplan}")
    event = self.controller.step(dict(action='GetReachablePositions'))
    self.reachable_positions = event.metadata['reachablePositions']
    return event

  def observation(self, event, task_id):
    task = np.zeros(self.ntasks, dtype=np.int32)
    task[task_id] = 1

    return dict(
      image=event.frame,
      task=task,
      )

  def reset(self):

    # -----------------------
    # every K tasks, reset floorplan (more efficient than always resetting)
    # -----------------------
    if self.tasks_in_floorplan >= self.tasks_per_floorplan_reset:
      event = self.reset_floorplan()
    self.tasks_in_floorplan += 1

    # -----------------------
    # sample new task object
    # -----------------------
    task_objects = []
    while len(task_objects) == 0:
      event = self.randomize_object_locations()
      event = self.randomize_agent()
      objects = event.metadata['objects']
      task_objects = list(filter(
              lambda o: o['objectType'] in self.tasks, objects))
      task_objects = list(filter(
              lambda o: o['distance'] >= self.min_task_dist and o['distance'] <=self.max_task_dist, task_objects))

    task_object = np.random.choice(task_objects)
    self.task_category =  task_object['objectType']
    self.task_id = self.object2idx[self.task_category]

    # -----------------------
    # reset task steps
    # -----------------------
    self.steps = 0

    return self.observation(event, self.task_id)

  def step(self, action):
    action_name = self.actions[action]
    event = self.controller.step(dict(action=action_name))
    obs = self.observation(event, self.task_id)
    info = dict()

    # ======================================================
    # check if task object found
    # ======================================================
    objects = event.metadata['objects']
    # objects that match category
    task_objects = list(filter(
            lambda o: o['objectType'] == self.task_category, objects))

    # objects within distance
    close_task_objects = list(filter(
            lambda o: o['distance'] <= self.success_distance, task_objects))

    # objects within distance + visible
    visible_task_objects = list(filter(
            lambda o: o['visible'], close_task_objects))

    # reward/done if found something
    reward = done = len(visible_task_objects) > 0
    info['success'] = done

    # increase distance task objects can be sampled
    if done:
      self.task_dist += self.task_dist_step

    # ======================================================
    # check if ran out of time
    # ======================================================
    self.steps += 1
    if not done:
      done = self.steps >= self.max_steps
    reward = float(reward)

    return obs, reward, done, info

  @property
  def max_task_dist(self):
    return self.min_task_dist + self.task_dist