import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.collections import namedarraytuple


from utils.ops import duplicate_vector

from nnmodules.structured_rnns import ListStructuredRnn
from nnmodules.transformer import attention

SchemasInput = namedarraytuple("SchemasInput", ["image", "task_action_reward"])  # For downstream namedarraytuples to work

TIME_AXIS=0
BATCH_AXIS=1
SCHEMA_AXIS=2

def compute_dims(num=None, total_dim=None, individual_dim=None):
  empty=[i is None for i in [num, total_dim, individual_dim]]

  if sum(empty) >=2:
    raise RuntimeError("Please provide at least 2 of `num`, `total_dim`, `individual_dim`")

  if num is None:
    num = total_dim//individual_dim
  elif total_dim is None:
    total_dim = num*individual_dim
  elif individual_dim is None:
    individual_dim = total_dim//num

  return num, total_dim, individual_dim

class MultiHeadedAttention(nn.Module):
    def __init__(self,
          num,
          dq,
          dk,
          d_out,
          dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.num = num
        self.Wq = nn.Linear(dq, d_out*num)
        self.Wk = nn.Linear(dk, d_out*num)
        self.Wv = nn.Linear(dk, d_out*num)
        self.out = nn.Linear(d_out*num, d_out)
        self.d_out = d_out
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_out => h x d_k 
        query = self.Wq(query).view(nbatches, -1, self.num, self.d_out)
        key = self.Wk(key).view(nbatches, -1, self.num, self.d_out)
        value = self.Wv(value).view(nbatches, -1, self.num, self.d_out)

        # 2) Apply attention on all the projected vectors in batch. 
        x, attn = attention(
          query.transpose(1, 2),
          key.transpose(1, 2),
          value.transpose(1, 2))
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num * self.d_out)

        return self.out(x), attn

class TopdownFeatureAttention(nn.Module):
    def __init__(self, query_dim, conv_dims, num_queries):
      super(TopdownFeatureAttention, self).__init__()

      self.channels, self.height, self.width = conv_dims

      self.linears = [nn.Linear(query_dim, self.channels) for _ in range(num_queries)]
      self.linears = torch.nn.ModuleList(self.linears)
      self.conv_out = nn.Conv2d(self.channels, self.channels, kernel_size=1)

    def forward(self, query, obs):
      # [T, B, N, D]
      T, B, N = query.shape[:3]
      C, H, W = obs.shape[-3:]
      weights = [lin(query[:, :, i]) for i, lin in enumerate(self.linears)]
      weights = torch.stack(weights, dim=SCHEMA_AXIS)
      weights = torch.sigmoid(weights)
      # [T, B, N, C, 1, 1]
      weights = weights.unsqueeze(4).unsqueeze(5)


      # [T, B, 1, C, H, W]
      obs_to_mod = obs.unsqueeze(2)


      modulated = obs_to_mod*weights
      out = self.conv_out(modulated.view(T*B*N, C, H, W))
      return out.view(T, B, N, C*H*W)


class PerceptulSchemas(nn.Module):
    """
    """
    def __init__(
      self,
      conv_dims,
      task_size,
      action_size,
      reward_size=1,
      schema_dim=None,
      total_dim=None,
      num_schemas=None,
      attn_heads=7,
      **kwargs
      ):
      """
      """
      super(PerceptulSchemas, self).__init__()
      save__init__args(locals())

      self.num_schemas, self.total_dim, self.schema_dim = compute_dims(num=num_schemas, total_dim=total_dim, individual_dim=schema_dim)

      self.context_dim = self.schema_dim + task_size + action_size + reward_size


      self.share_info = MultiHeadedAttention(
        dq=self.context_dim,
        dk=self.schema_dim,
        d_out=self.schema_dim,
        num=attn_heads)

      self.topdown = TopdownFeatureAttention(
        query_dim=self.context_dim,
        conv_dims=conv_dims,
        num_queries=self.num_schemas)

      mem_input_size = int(np.prod(conv_dims)) # update
      mem_input_size += self.schema_dim        # sharing info
      mem_input_size += self.context_dim       # rest
      self.lstms = ListStructuredRnn(
              num=self.num_schemas,
              input_size=mem_input_size,
              hidden_size=self.schema_dim)


    def forward(self, x: SchemasInput, prev_state=None):
      image = x.image # [T, B, C, H, W]
      task_action_reward = x.task_action_reward # [T, B, D]
      T, B = image.shape[:2]
      N = self.num_schemas

      # [T, B, N, D]
      if prev_state is None:
        prev_h = torch.zeros(T, B, N, self.schema_dim)
      else:
        (prev_h, prev_c) = prev_state
        prev_h = prev_h.view(1, B, N, self.schema_dim)
        if T > 1:
          import ipdb; ipdb.set_trace()


      # [T, B, N, D]
      schema_context = duplicate_vector(task_action_reward, N, dim=SCHEMA_AXIS)
      schema_context = torch.cat((schema_context, prev_h), dim=-1)

      # ======================================================
      # attention to other schemas
      # ======================================================
      zeros=torch.zeros(T, B, 1, self.schema_dim)
      keys = torch.cat((prev_h, zeros), dim=SCHEMA_AXIS)
      shared, attn = self.share_info(
        query=schema_context.view(T*B, N, -1),
        key=keys.view(T*B, N+1, -1),
        value=keys.view(T*B, N+1, -1))

      # [T, B, N, D]
      shared = shared.view(T, B, N, -1)

      # ======================================================
      # top-down attention over observation
      # ======================================================
      # [T, B, N, C*H*W]
      schema_updates = self.topdown(
        query=schema_context,
        obs=image)

      # ======================================================
      # Update LSTMs
      # ======================================================
      # [T, B, N, D]
      mem_input = torch.cat((schema_updates, shared, schema_context), dim=-1)
      # [T, B, N*D]
      outs, (hs, cs) = self.lstms(mem_input, prev_state)

      outs = outs.view(T, B, -1)
      return outs, (hs, cs)

      # ======================================================
      # Return Outputs
      # ======================================================