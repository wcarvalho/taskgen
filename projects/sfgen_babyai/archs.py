import torch
import torch.nn as nn
from nnmodules.perceptual_schemas import PerceptulSchemas
from nnmodules.sfgen_modelv2 import SFGenModelBase, lstm_input_fn

def struct_mem_input_fn(image, task, action, reward):
  return image, torch.cat([
            task,
            action,
            prev_reward,
            ], dim=2)

class LstmDqn(SFGenModelBase):
  """docstring for LstmDqn"""
  def __init__(self,
      **kwargs):
    kwargs.pop('MemoryCls', None)
    kwargs.pop('memory_input_fn', None)
    kwargs.pop('rlhead', None)
    super().__init__(
      MemoryCls=nn.LSTM,
      memory_input_fn=lstm_input_fn,
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
    kwargs.pop('rlhead', None)
    super().__init__(
      MemoryCls=nn.LSTM,
      memory_input_fn=lstm_input_fn,
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
