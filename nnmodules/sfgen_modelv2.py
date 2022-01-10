from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

from nnmodules.perceptual_schemas import PerceptulSchemas
from nnmodules.babyai_model import BabyAIModel, DQNHead
from utils.ops import duplicate_vector
from utils.ops import check_for_nan_inf
RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work

def lstm_input_fn(image, task, action, reward, T, B):
  return torch.cat([
    image.view(T, B, -1),
    task.view(T, B, -1),
    action.view(T, B, -1),
    reward.view(T, B, 1),
    ], dim=2)

def dqn_input_size(image_size, task_size, action_size, reward_size):
  # task is given to lstm typically
  return image_size+task_size+action_size+reward_size


class DqnModelBase(BabyAIModel):
    """
    """
    def __init__(
        self,
        output_size,
        MemoryCls=nn.LSTM,
        memory_kwargs=None,
        memory_input_fn=lstm_input_fn,
        memory_input_size=dqn_input_size,
        task_size=128,
        head_size=None,
        hidden_policy_layers=0,
        gvf_size=None,
        default_size=None,
        rlhead='gvf',
        **kwargs
        ):
        """
        """
        kwargs.pop("text_output_size", None)
        super(DqnModelBase, self).__init__(
          # text_output_size=task_size,
          **kwargs)
        # optionally keep everything same dimension and just scale
        head_size = default_size if head_size is None else head_size
        gvf_size = default_size if gvf_size is None else gvf_size

        save__init__args(locals())
        if self.pretrained_embeddings > 0:
          self.task_size = task_size = self.pretrained_embeddings

        # -----------------------
        # Memory
        # -----------------------
        self.memory = self.build_memory()

        if rlhead == 'gvf':
            self.rl_head = DqnGvfHead(
                input_size=self.state_size + task_size,
                gvf_size=gvf_size,
                state_size=self.state_size,
                head_size=head_size,
                hidden_layers=hidden_policy_layers,
                num_actions=output_size,
                task_size=self.task_size,
                nonlinearity=self.nonlinearity_fn,
                )
        elif rlhead == 'dqn':
            self.rl_head = DQNHead(
                input_size=self.state_size + task_size,
                head_size=head_size,
                hidden_layers=hidden_policy_layers,
                output_size=output_size,
                dueling=False,
                )
        else:
            raise RuntimeError(f"Unsupported:'{rlhead}'")

    def build_memory(self):
      self.memory_kwargs = self.memory_kwargs or dict()
      conv_flat = int(np.prod(self.conv.output_dims))
      self.memory_kwargs['input_size'] = self.memory_input_size(
        image_size=conv_flat,
        task_size=self.task_size,
        action_size=self.output_size,
        reward_size=1)

      return self.MemoryCls(**self.memory_kwargs)


    def forward(self, observation, prev_action, prev_reward, init_rnn_state, done=None, all_variables=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        variables=dict()
        lead_dim, T, B, img_shape = infer_leading_dims(observation.image, 3)

        # ======================================================
        # Vision
        # ======================================================
        image_embedding, mission_embedding, direction_embedding = self.process_observation(observation)
        if all_variables:
          variables['image_embedding'] = image_embedding
          variables['mission_embedding'] = mission_embedding
          variables['direction_embedding'] = direction_embedding


        # ======================================================
        # Memory
        # ======================================================
        memory_input = self.memory_input_fn(
            image=image_embedding,
            task=mission_embedding,
            action=prev_action,
            reward=prev_reward,
            T=T, B=B,
          )
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        state, (hn, cn) = self.memory(memory_input, init_rnn_state)
        next_rnn_state = RnnState(h=hn, c=cn)
        if all_variables:
          variables['state'] = state

        # ======================================================
        # Policy
        # ======================================================
        if all_variables:
            self.rl_head([state], mission_embedding, 
                final_fn=partial(restore_leading_dims, lead_dim=lead_dim, T=T, B=B),
                variables=variables)
            return variables
        else:
            rl_out = self.rl_head([state], mission_embedding)
            # Restore leading dimensions: [T,B], [B], or [], as input.
            rl_out = restore_leading_dims(rl_out, lead_dim, T, B)
            return list(rl_out) + [next_rnn_state]

    @property
    def state_size(self):
      raise NotImplementedError


class SFGenModelBase(DqnModelBase):

  def build_memory(self):
    self.memory_kwargs = self.memory_kwargs or dict()
    self.memory_kwargs.update(
      conv_dims=self.conv.output_dims,
      task_size=self.task_size,
      action_size=self.output_size,
      reward_size=1)

    memory = self.MemoryCls(**self.memory_kwargs)
    return memory


class DqnGvfHead(torch.nn.Module):
    """docstring for DQNHead"""
    def __init__(self, input_size, gvf_size, state_size, hidden_layers, head_size, num_actions, task_size, nonlinearity=torch.nn.ReLU, **kwargs):
        super(DqnGvfHead, self).__init__()
        self.head_size = head_size
        self.num_actions = num_actions
        self.state_size = state_size


        # 1-layer MLP (2 linear)
        self.gvf = MlpModel(
          input_size=input_size,
          hidden_sizes=[gvf_size] if gvf_size else [],
          output_size=num_actions*state_size,
          nonlinearity=nonlinearity,
          )

        # Linear layer 
        hidden_sizes=[head_size]*hidden_layers if hidden_layers > 0 else []
        self.successor_head = MlpModel(state_size,
          hidden_sizes=hidden_sizes,
          output_size=head_size)

        self.task_weights = nn.Linear(task_size, head_size)

    def forward(self, state_variables, task, final_fn=lambda x:x, variables=None):
        """
        """
        T, B = task.shape[:2]
        A = self.num_actions

        state_variables.append(task)
        gvf_input = torch.cat(state_variables, dim=-1)

        # T x B x |A|*D
        predictive_state = self.gvf(gvf_input)
        predictive_state = predictive_state.view(T, B, A, self.state_size)
        if variables is not None:
          variables['predictive_state'] = predictive_state

        # TBA x H
        successor_features = self.successor_head(predictive_state.view(T*B*A, -1))
        # T X B X A X H
        successor_features = successor_features.view(T, B, A, self.head_size)

        # T X B X H
        weights = self.task_weights(task)
        # T X B X 1 X H
        weights = weights.unsqueeze(2)
        # T X B X H X 1 (for dot-product)
        weights = weights.transpose(-2, -1)

        # dot product
        q_values = torch.matmul(successor_features, weights).squeeze(-1)

        q_values = q_values.view(T*B, -1)
        q_values = final_fn(q_values)
        check_for_nan_inf(q_values)

        if variables is not None:
          variables['q'] = q_values
        else:
            return [q_values]

