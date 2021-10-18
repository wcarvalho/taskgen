import torch
import torch.nn as nn
from nnmodules.perceptual_schemas import PerceptulSchemas, SchemasInput
from nnmodules.sfgen_modelv2 import SFGenModelBase, lstm_input_fn, dqn_input_size


def struct_mem_input_fn(image, task, action, reward):
  return SchemasInput(image, torch.cat([
            task,
            action,
            reward,
            ], dim=2))

def gvf_mem_input_fn(image, task, action, reward):
  return torch.cat([
            image,
            action,
            reward,
            ], dim=2)

def struct_mem_input_size(image_size, task_size, action_size, reward_size):
  return image_size+task_size+action_size+reward_size


def gvf_input_size(image_size, task_size, action_size, reward_size):
  # task is given to head 
  return image_size+action_size+reward_size

class LstmDqn(SFGenModelBase):
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



class LstmGvf(SFGenModelBase):
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



class SchemasGvf(SFGenModelBase):
  """docstring for LstmDqn"""
  def __init__(self,
      **kwargs):
    kwargs.pop('MemoryCls', None)
    kwargs.pop('memory_input_fn', None)
    kwargs.pop('memory_input_size', None)
    kwargs.pop('rlhead', None)
    super().__init__(
      MemoryCls=functool.partial(
        PerceptulSchemas,
        task_size=kwargs['task_size'],
        ),
      memory_input_fn=struct_mem_input_fn,
      memory_input_size=struct_mem_input_size,
      rlhead='gvf',
      **kwargs
      )

  @property
  def state_size(self):
    return self.memory_kwargs['total_size']
