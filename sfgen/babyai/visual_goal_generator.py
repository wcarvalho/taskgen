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
from sfgen.babyai.structured_rnns import ListStructuredRnn

class VisualGoalGenerator(nn.Module):
    """
    1. Compute modulation on conv features
    2. apply to conv features
    3. compress conv features
    """
    def __init__(self,
        task_dim,
        goal_dim,
        pre_mod_layer,
        conv_feature_dims,
        mod_function,
        mod_compression,
        goal_tracking,
        use_history,
        nonlinearity=torch.nn.ReLU,
        nheads=4,
        ):
        super(VisualGoalGenerator, self).__init__()
        save__init__args(locals())

        channels, height , width = conv_feature_dims

        self.modulation_generator = ModulationGenerator(
            task_dim=task_dim,
            pre_mod_layer=pre_mod_layer,
            goal_dim=goal_dim,
            use_history=use_history,
            conv_feature_dims=conv_feature_dims,
            mod_function=mod_function,
            nonlinearity=nonlinearity,
            nheads=nheads,
            )

        if mod_compression == 'maxpool':
            self.compression = nn.MaxPool2d(kernel_size=(height, width), stride=2)
            self.goal_dim = channels
        elif mod_compression == 'avgpool':
            self.compression = nn.AvgPool2d(kernel_size=(height, width), stride=2)
            self.goal_dim = channels
        elif mod_compression == 'linear':
            self.compression = nn.Linear(channels*height*width, self.goal_dim)
        else:
            raise NotImplementedError

        if use_history:
            raise NotImplementedError("Need a custom cell for time-series that takes initalization and loops. little bit of work. not done yet.")

        if goal_tracking == 'sum':
            raise NotImplementedError
            # self.goal_tracker = SumGoalHistory(self.goal_dim)
        elif goal_tracking == 'lstm':
            self.goal_tracker = ListStructuredRnn(nheads, self.goal_dim, self.goal_dim)
        else:
            raise NotImplementedError


    @property
    def hist_dim(self):
        return self.goal_dim // self.nheads
    
    @property
    def output_dim(self):
        return self.goal_dim

    def forward(self, obs_emb, task_emb, init_goal_state=None):
        T, B = obs_emb.shape[:2]
        # if init_goal_state is None:
        #     assert T == 1, "shouldn't happen in T > 1 case"
        #     zeros = torch.zeros((T, B, self.goal_dim), device=task_emb.device, dtype=task_emb.dtype)
        #     init_goal_state = (zeros, zeros)

        # T x B x H x C
        modulation_weights = self.modulation_generator(task_emb, init_goal_state)

        # T x B x H x C x 1 x 1
        modulation_weights = modulation_weights.unsqueeze(4).unsqueeze(5)

        # T x B x H x C x N x N
        obs_to_mod = obs_emb.unsqueeze(2)

        modulated = obs_to_mod*modulation_weights

        # checks
        # (modulation_weights[:,:,0]*obs_emb == modulated[:,:,0]).all()


        if 'pool' in self.mod_compression:
            goal = self.compression(modulated.view(T*B, *modulated.shape[2:]))
            goal = goal.view(T, B, -1)
        elif self.mod_compression == 'linear':
            goal = self.compression(modulated.view(T, B, self.nheads, -1))
        else:
            raise NotImplementedError

        out, (h, c) = self.goal_tracker(goal, init_goal_state)


        return goal, out, (h, c)


class ModulationGenerator(nn.Module):
    """docstring for ModulationGenerator"""
    def __init__(self,
        task_dim,
        conv_feature_dims,
        use_history=False,
        goal_dim=0,
        pre_mod_layer=False,
        mod_function='sigmoid',
        nonlinearity=torch.nn.ReLU,
        nheads=4,
        ):
        super(ModulationGenerator, self).__init__()
        save__init__args(locals())

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
                    nonlinearity=nonlinearity
                    )
            else:
                self.task_prelayer = lambda x:x
        else:
            dim = task_dim
        self.weight_generator = nn.Linear(dim, nheads*channels)


    def forward(self, task_emb, goal_state):
        """
        get modulation based on task and goal history
        """
        T, B = task_emb.shape[:2]

        if self.use_history:
            raise NotImplemented("Will need a custom generation/tracking cell for this")
            task_embed = torch.cat((task_emb, goal_state), dim=-1)
            task_embed = self.task_prelayer(task_embed)
        else:
            task_embed = task_emb


        weights = self.weight_generator(task_embed)
        weights = weights.view(T, B, self.nheads, self.conv_feature_dims[0])

        if self.mod_function.lower() == 'sigmoid':
            weights = torch.sigmoid(weights)
        elif self.mod_function.lower() == 'norm':
            weights = F.normalize(weights, p=2, dim=-1)
        elif self.mod_function.lower() == 'none':
            pass
        else:
            raise NotImplementedError

        return weights
















