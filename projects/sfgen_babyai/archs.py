import functools
import torch
import torch.nn as nn
from nnmodules.perceptual_schemas import PerceptulSchemas, SchemasInput
from nnmodules.sfgen_modelv2 import SFGenModelBase, DqnModelBase,lstm_input_fn, dqn_input_size


def struct_mem_input_fn(image, task, action, reward, T, B):
  return SchemasInput(image, torch.cat([
            task.view(T, B, -1),
            action.view(T, B, -1),
            reward.view(T, B, 1),
            ], dim=2))

def gvf_mem_input_fn(image, task, action, reward, T, B):
  return torch.cat([
            image.view(T, B, -1),
            action.view(T, B, -1),
            reward.view(T, B, 1),
            ], dim=2)

def struct_mem_input_size(image_size, task_size, action_size, reward_size):
  return image_size+task_size+action_size+reward_size


def gvf_input_size(image_size, task_size, action_size, reward_size):
  # task is given to head 
  return image_size+action_size+reward_size

class LstmDqn(DqnModelBase):
  """docstring for LstmDqn"""
  def __init__(self,
      **kwargs):
    kwargs.pop('MemoryCls', None)
    kwargs.pop('memory_input_fn', None)
    kwargs.pop('memory_input_size', None)
    kwargs.pop('rlhead', None)
    super().__init__(
      MemoryCls=nn.LSTM,
      memory_input_fn=lstm_input_fn,
      memory_input_size=dqn_input_size,
      rlhead='dqn',
      **kwargs
      )

  @property
  def state_size(self):
    return self.memory_kwargs['hidden_size']



class LstmGvf(DqnModelBase):
  """docstring for LstmDqn"""
  def __init__(self,
      **kwargs):
    kwargs.pop('MemoryCls', None)
    kwargs.pop('memory_input_fn', None)
    kwargs.pop('memory_input_size', None)
    kwargs.pop('rlhead', None)
    super().__init__(
      MemoryCls=nn.LSTM,
      memory_input_fn=gvf_mem_input_fn,
      memory_input_size=gvf_input_size,
      rlhead='gvf',
      **kwargs
      )


  @property
  def state_size(self):
    return self.memory_kwargs['hidden_size']

class SchemasDqn(SFGenModelBase):
  """docstring for LstmDqn"""
  def __init__(self,
      **kwargs):
    kwargs.pop('MemoryCls', None)
    kwargs.pop('memory_input_fn', None)
    kwargs.pop('memory_input_size', None)
    kwargs.pop('rlhead', None)
    super().__init__(
      MemoryCls=PerceptulSchemas,
      memory_input_fn=struct_mem_input_fn,
      rlhead='dqn',
      **kwargs
      )

  @property
  def state_size(self):
    return self.memory_kwargs['total_dim']


class SchemasGvf(SFGenModelBase):
  """docstring for LstmDqn"""
  def __init__(self,
      **kwargs):
    kwargs.pop('MemoryCls', None)
    kwargs.pop('memory_input_fn', None)
    kwargs.pop('memory_input_size', None)
    kwargs.pop('rlhead', None)
    super().__init__(
      MemoryCls=PerceptulSchemas,
      memory_input_fn=struct_mem_input_fn,
      rlhead='gvf',
      **kwargs
      )

  @property
  def state_size(self):
    return self.memory_kwargs['total_dim']
