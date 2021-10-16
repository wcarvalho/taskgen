from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import models, transforms


from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.quick_args import save__init__args

from nnmodules.modules import initialize_parameters



RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work



class Resnet(torch.nn.Module):
    # copied from: https://github.com/askforalfred/alfred/blob/master/models/nn/resnet.py

    def __init__(self, use_conv_feat=True):
      super().__init__()

      self.model = models.resnet18(pretrained=True)
      # import ipdb; ipdb.set_trace()
      if use_conv_feat:
          self.model = nn.Sequential(*list(self.model.children())[:-2])
      self.model = self.model.eval()


    def forward(self, images):
        return self.model(images)

    @property
    def output_size(self):
      return np.prod(self.output_dims)

    @property
    def output_dims(self):
      return (512, 7, 7)



class ThorModel(torch.nn.Module):
    """
    """
    def __init__(
            self,
            image_shape,
            task_shape,  # discrete task
            output_size, # actions
            use_conv_feat=True,
            fc_size=512,
            head_size=256,
            lstm_size=512,
            task_size=128,
            action_size=64,
            rlhead='ppo',
            **kwargs,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        save__init__args(locals())

        assert image_shape[0] >= 224 and image_shape[1] >= 224

        # -----------------------
        # vision model
        # -----------------------
        self.model = Resnet(use_conv_feat=use_conv_feat)
        for param in self.model.parameters():
            param.requires_grad = False
        # -----------------------
        # LSTM
        # -----------------------
        self.lstm = torch.nn.LSTM(self.model.output_size + action_size + task_size + 1, lstm_size)

        # -----------------------
        # policy
        # -----------------------
        input_size = lstm_size + task_size
        if rlhead == 'dqn':
            self.rl_head = DQNHead(
                input_size=input_size,
                head_size=head_size,
                output_size=output_size,
                dueling=dueling)
        elif rlhead == 'ppo':
            self.rl_head = PPOHead(
                input_size=input_size, 
                output_size=output_size,
                hidden_size=head_size)
            self.apply(initialize_parameters)
        else:
            raise RuntimeError(f"RL Algorithm '{rlhead}' unsupported")

        # -----------------------
        # task embedding
        # -----------------------
        self.task_embedding = torch.nn.Embedding(
          num_embeddings=task_shape[0],
          embedding_dim=task_size,
          )

        self.action_embedding = torch.nn.Embedding(
          num_embeddings=output_size,
          embedding_dim=action_size,
          )

    def process_observation(self, observation):
        """
        """

        # ======================================================
        # Process image
        # ======================================================
        # process image
        img = observation.image
        img = img.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        with torch.set_grad_enabled(False):
          image_embedding = self.model(img.view(T * B, *img_shape))
        image_embedding = image_embedding.view(T, B, *self.model.output_dims)

        task = observation.task.argmax(-1).view(T*B)
        task_embedding = self.task_embedding(task)
        task_embedding = task_embedding.view(T, B, -1)

        return image_embedding, task_embedding

    def forward(self, observation, prev_action, prev_reward, init_rnn_state, all_variables=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        variables=dict()
        lead_dim, T, B, img_shape = infer_leading_dims(observation.image, 3)

        # ======================================================
        # Process obs
        # ======================================================
        image_embedding, task_embedding = self.process_observation(observation)
        action = prev_action.argmax(-1).view(T*B)
        action_embedding = self.action_embedding(action)
        action_embedding = action_embedding.view(T, B, -1)

        if all_variables:
            variables['image_embedding'] = image_embedding
            variables['task_embedding'] = task_embedding
            variables['action_embedding'] = action_embedding

        # ======================================================
        # LSTM
        # ======================================================
        lstm_input = torch.cat([
            image_embedding.view(T, B, -1),
            task_embedding.view(T, B, -1),
            action_embedding.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)

        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        next_rnn_state = RnnState(h=hn, c=cn)
        if all_variables:
            variables['lstm_out'] = lstm_out


        if all_variables:
            self.rl_head([lstm_out], task_embedding, 
                final_fn=partial(restore_leading_dims, lead_dim=lead_dim, T=T, B=B),
                variables=variables)
            return variables
        else:
            rl_out = self.rl_head([lstm_out], task_embedding)
            # Restore leading dimensions: [T,B], [B], or [], as input.
            rl_out = restore_leading_dims(rl_out, lead_dim, T, B)

            return list(rl_out) + [next_rnn_state]



class PPOHead(torch.nn.Module):
    """
    """
    def __init__(self, input_size, output_size, hidden_size=64):
        super(PPOHead, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.pi = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, output_size)
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, state_variables, task, final_fn=lambda x:x, variables=None):
        """
        """
        state_variables.append(task)
        state = torch.cat(state_variables, dim=-1)

        T, B = state.shape[:2]
        pi = F.softmax(self.pi(state.view(T * B, -1)), dim=-1)
        v = self.value(state.view(T * B, -1)).squeeze(-1)

        pi = final_fn(pi)
        v = final_fn(v)
        if variables is not None:
            variables['pi'] = pi
            variables['v'] = v
        else:
            return [pi, v]



class DQNHead(torch.nn.Module):
    """docstring for DQNHead"""
    def __init__(self, input_size, head_size, output_size, dueling):
        super(DQNHead, self).__init__()
        self.dueling = dueling

        if dueling:
            self.head = DuelingHeadModel(input_size, head_size, output_size)
        else:
            self.head = MlpModel(input_size, head_size, output_size=output_size)

    def forward(self, state_variables, task, final_fn=lambda x:x, variables=None):
        """
        """
        state_variables.append(task)
        state = torch.cat(state_variables, dim=-1)
        T, B = state.shape[:2]
        q = self.head(state.view(T * B, -1))
        q = final_fn(q)

        if variables is not None:
            variables['q'] = q
        else:
            return [q]
