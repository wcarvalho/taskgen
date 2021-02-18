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
        history_dim,
        pre_mod_layer,
        conv_feature_dims,
        mod_function,
        mod_compression,
        goal_tracking,
        use_history,
        nonlinearity=torch.nn.ReLU,
        nheads=4,
        normalize_goal=False,
        independent_compression=False,
        ):
        super(VisualGoalGenerator, self).__init__()
        save__init__args(locals(), underscore=True)

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

        if 'pool' in mod_compression:
            raise NotImplementedError
        # else:
            # self.total_dim = goal_dim
            # self.head_dim = self.total_dim // nheads



        if mod_compression == 'maxpool':
            raise NotImplementedError
            # self.compression = nn.MaxPool2d(kernel_size=(height, width), stride=2)
            # self.goal_dim = channels
        elif mod_compression == 'avgpool':
            raise NotImplementedError
            # self.compression = nn.AvgPool2d(kernel_size=(height, width), stride=2)
            # self.goal_dim = channels
        elif mod_compression == 'linear':
            if independent_compression:
                self._goal_dim = goal_dim//nheads
                self.compression = nn.ModuleList([
                    nn.Linear(channels*height*width, self.goal_dim) for _ in range(nheads)])
            else:
                self.compression = nn.Linear(channels*height*width, self.goal_dim)
        else:
            raise NotImplementedError

        if use_history:
            raise NotImplementedError("Need a custom cell for time-series that takes initalization and loops. little bit of work. not done yet.")

        if goal_tracking == 'sum':
            raise NotImplementedError
            # self.goal_tracker = SumGoalHistory(self.goal_dim)
        elif goal_tracking == 'lstm':
            self.goal_tracker = ListStructuredRnn(
                num=self._nheads,
                input_size=self._goal_dim,
                hidden_size=self._history_dim)
        else:
            raise NotImplementedError



    @property
    def hist_dim(self):
        return self._history_dim // self._nheads

    @property
    def goal_dim(self):
        return self._goal_dim

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

        # T x B x 1 x C x N x N
        obs_to_mod = obs_emb.unsqueeze(2)

        modulated = obs_to_mod*modulation_weights

        # checks
        # (modulation_weights[:,:,0]*obs_emb == modulated[:,:,0]).all()


        if 'pool' in self._mod_compression:
            raise NotImplementedError
            # goal = self.compression(modulated.view(T*B, *modulated.shape[2:]))
            # goal = goal.view(T, B, -1)
        elif self._mod_compression == 'linear':
            if self._independent_compression:
                modulations = modulated.view(T, B, self._nheads, -1)
                goal = []
                for idx in range(self._nheads):
                    goal.append(self.compression[idx](modulations[:,:, idx]))
                goal = torch.stack(goal, dim=2)
            else:
                goal = self.compression(modulated.view(T, B, self._nheads, -1))
        else:
            raise NotImplementedError

        if self._normalize_goal:
            goal = F.normalize(goal + 1e-12, p=2, dim=-1)

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
















