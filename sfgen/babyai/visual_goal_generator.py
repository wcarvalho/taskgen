import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.quick_args import save__init__args
from typing import List, Tuple

import torch
import torch.jit as jit

from sfgen.babyai.modules import BabyAIFiLMModulation, GatedModulation

class VisualGoalGenerator(nn.Module):
    """docstring for VisualGoalGenerator"""
    def __init__(self,
        task_dim,
        goal_dim,
        pre_mod_layer,
        conv_feature_dims,
        mod_function,
        mod_compression,
        goal_tracking,
        use_history,
        ):
        super(VisualGoalGenerator, self).__init__()
        save__init__args(locals())

        channels, height , width = conv_feature_dims
        self.mod_compression = mod_compression
        if mod_compression == 'maxpool':
            self.compression = nn.MaxPool2d(kernel_size=(height, width), stride=2)
        elif mod_compression == 'avgpool':
            self.compression = nn.AvgPool2d(kernel_size=(height, width), stride=2)
        elif mod_compression == 'linear':
            self.compression = nn.Linear(channels*height*width, channels)
        else:
            raise NotImplementedError

        if use_history:
            raise NotImplementedError("Need a custom cell for time-series that takes initalization and loops. little bit of work. not done yet.")

        self.modulation_generator = ModulationGenerator(
            task_dim=task_dim,
            pre_mod_layer=pre_mod_layer,
            goal_dim=goal_dim,
            use_history=use_history,
            conv_feature_dims=conv_feature_dims,
            mod_function=mod_function,
            )

        if goal_tracking == 'sum':
            self.goal_tracker = SumGoalHistory(goal_dim)
        elif goal_tracking == 'lstm':
            self.goal_tracker = nn.LSTM(goal_dim, goal_dim)
        else:
            raise NotImplementedError


    def forward(self, obs_emb, task_emb, init_goal_state=None):
        T, B = obs_emb.shape[:2]
        if init_goal_state is None:
            assert T == 1, "shouldn't happen in T > 1 case"
            zeros = torch.zeros((T, B, self.goal_dim), device=task_emb.device, dtype=task_emb.dtype)
            init_goal_state = (zeros, zeros)

        modulation_weights = self.modulation_generator(task_emb, init_goal_state[0])

        modulated = obs_emb*modulation_weights.unsqueeze(-1).unsqueeze(-1)

        if 'pool' in self.mod_compression:
            goal = self.compression(modulated.view(T*B, *modulated.shape[2:]))
            goal = goal.view(T, B, -1)
        elif self.mod_compression == 'linear':
            goal = self.compression(modulated.view(T, B, -1))
        else:
            raise NotImplementedError

        out, (h, c) = self.goal_tracker(goal, init_goal_state)

        return goal, out, (h, c)


class SumGoalHistory(nn.Module):
    """docstring for SumGoalHistory"""
    def __init__(self, output_function='sigmoid'):
        super(SumGoalHistory, self).__init__()
        self.output_function = output_function


    @jit.script_method
    def process(self, goal, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> (Tensor, Tensor)
        inputs = input.unbind(0)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):

            out = state = state + goal

            if self.output_function.lower() == 'sigmoid':
                out = torch.sigmoid(out)
            elif self.output_function.lower() == 'norm':
                out = F.normalize(out, p=2, dim=-1)
            elif self.output_function.lower() == 'none':
                pass
            else:
                raise NotImplementedError
            outputs += [out]

        return torch.stack(outputs), state

    def forward(self, goal:torch.Tensor, init_goal_state:torch.Tensor=None):
        """Squeeze/unsqueeze data so follows convention of pytorch's LSTM class
        
        if state is None, it's constructed.
        if task has only length=1 along time-dimension, same value is used at every time-step.
        
        Args:
            goal (torch.Tensor): T x B x D
            init_goal_state (torch.Tensor, optional): 1 x B x D
        
        Deleted Parameters:
            input (TYPE): T x B x D
            state (TYPE): 1 x B x D or None
            task (TYPE): T x B x D or 1 x B x D
        
        Returns:
            TYPE: Description
        
        Raises:
            NotImplementedError: Description
        
        """

        T, B, D = goal.shape
        if init_goal_state is None:
            init_state = torch.zeros(1, B, D, device=goal.device, type=gaol.type)
        else:
            init_state = init_goal_state[0]

        outputs, state = self.process(goal, init_state)
        if T > 1 and B > 1:
            import ipdb; ipdb.set_trace()


        return outputs, (state.unsqueeze(0), state.unsqueeze(0))



class ModulationGenerator(nn.Module):
    """docstring for ModulationGenerator"""
    def __init__(self,
        task_dim,
        conv_feature_dims,
        use_history=False,
        goal_dim=0,
        pre_mod_layer=False,
        mod_function='sigmoid',
        ):
        super(ModulationGenerator, self).__init__()

        self.use_history = use_history
        self.mod_function = mod_function
        channels, height , width = conv_feature_dims

        # ======================================================
        # task embedding
        # ======================================================
        if self.use_history:
            dim = task_dim+goal_dim
            if pre_mod_layer:
                self.task_prelayer = MlpModel(
                    input_size=dim,
                    output_size=dim,
                    nonlinearity=torch.nn.ReLU
                    )
            else:
                self.task_prelayer = lambda x:x
        else:
            dim = goal_dim
        self.task_linear = nn.Linear(dim, channels)


    def forward(self, task_emb, goal_state):
        """
        get modulation based on task and goal history
        """
        T, B = task_emb.shape[:2]

        if self.use_history:
            task_embed = torch.cat((task_emb, goal_state), dim=-1)
            task_embed = self.task_prelayer(task_embed)
        else:
            task_embed = task_emb
        weights = self.task_linear(task_embed)

        if self.mod_function.lower() == 'sigmoid':
            weights = torch.sigmoid(weights)
        elif self.mod_function.lower() == 'norm':
            weights = F.normalize(weights, p=2, dim=-1)
        elif self.mod_function.lower() == 'none':
            pass
        else:
            raise NotImplementedError

        return weights




