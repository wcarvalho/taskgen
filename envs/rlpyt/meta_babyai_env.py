"""
Class used to define rlpyt compatible meta-environment for `babyai` and `babyai_kitchen`. This class takes in a list of 
"""

import copy

import gym
import numpy as np
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import namedarraytuple, namedtuple
from envs.babyai_kitchen.levelgen import KitchenLevel

EnvInfo = namedtuple("EnvInfo", ['success'])  # Define in env file.

PixelObservation = namedarraytuple("PixelObservation", ["image", "mission", "mission_idx"])
SymbolicObservation = namedarraytuple("SymbolicObservation", ["image", "mission", "mission_idx", "direction"])
