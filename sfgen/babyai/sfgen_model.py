from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

from sfgen.babyai.modules import initialize_parameters, ObservationLSTM
from sfgen.babyai.visual_goal_generator import VisualGoalGenerator
from sfgen.babyai.babyai_model import BabyAIModel
from sfgen.tools.ops import duplicate_vector
from sfgen.tools.ops import check_for_nan_inf
RnnState = namedarraytuple("RnnState", ["h_obs", "c_obs", "h_goal", "c_goal"])

class SFGenModel(BabyAIModel):
    """
    """
    def __init__(
        self,
        output_size,
        pre_mod_layer=False, # whether to apply layer to goal/task before obtaining cnn weights
        obs_in_state=True,
        goal_use_history=False,
        normalize_history=False,
        mod_function='sigmoid',
        mod_compression='maxpool',
        goal_tracking='lstm',
        goal_size=256,
        lstm_size=256,
        head_size=256,
        gvf_size=256,
        default_size=0,
        obs_fc_size=256,
        dueling=False,
        rlhead='dqn',
        **kwargs
        ):
        """
        """
        super(SFGenModel, self).__init__(**kwargs)
        # optionally keep everything same dimension and just scale
        if default_size > 0:
            goal_size = default_size
            lstm_size = default_size
            head_size = default_size
            gvf_size = default_size
            obs_fc_size = default_size

        save__init__args(locals())
        assert dueling == False, "Successor doesn't support dueling currently"


        self.observation_memory = ObservationLSTM(
            conv_feature_dims=self.conv.output_dims,
            lstm_size=lstm_size,
            fc_size=obs_fc_size,
            # action dim + reward + direction
            extra_input_dim=output_size+1+self.direction_embed_size,
        )

        # goal_dim = self.conv.output_dims[0] # number of channels
        task_dim = self.text_embed_size
        self.goal_generator = VisualGoalGenerator(
            conv_feature_dims=self.conv.output_dims,
            task_dim=task_dim,
            goal_dim=goal_size, # number of channels
            pre_mod_layer=pre_mod_layer,
            mod_function=mod_function,
            mod_compression=mod_compression,
            goal_tracking=goal_tracking,
            use_history=goal_use_history,
            nonlinearity=self.nonlinearity_fn,
        )


        self.goal_gvf = MlpModel(input_size=2*self.goal_generator.output_dim + task_dim,
            hidden_sizes=[gvf_size] if gvf_size else [],
            output_size=output_size*head_size,
            nonlinearity=self.nonlinearity_fn,
            )

        # self.goal_prediction_head = MlpModel(input_size, head_size, output_size=head_size*output_size)
        # successor_features = self.successor_head(state.view(T*B, -1)).view(T*B, self.output_size, self.head_size)


        if obs_in_state:
            input_size = head_size + lstm_size
        else:
            input_size = head_size

        if rlhead == 'dqn':
            self.rl_head = DqnGvfHead(
                input_size=input_size,
                head_size=head_size,
                output_size=output_size,
                task_dim=self.text_embed_size,
                nonlinearity=self.nonlinearity_fn,
                )
        elif rlhead == 'ppo':
            raise NotImplementedError("PPO")

        else:
            raise RuntimeError(f"Unsupported:'{rlhead}'")


    def forward(self, observation, prev_action, prev_reward, init_rnn_state, all_variables=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        variables=dict()
        lead_dim, T, B, img_shape = infer_leading_dims(observation.image, 3)

        image_embedding, mission_embedding, direction_embedding = self.process_observation(observation)


        variables['image_embedding'] = image_embedding
        variables['mission_embedding'] = mission_embedding
        variables['direction_embedding'] = direction_embedding
        # ======================================================
        # pass CNN output through Observation LSTM
        # ======================================================

        non_obs_inputs = [e for e in [direction_embedding] if e is not None]
        non_obs_inputs.extend([prev_action, prev_reward])
        obs_mem_outputs, (h_obs, c_obs) = self.observation_memory(
            obs_emb=image_embedding,
            init_lstm_inputs=non_obs_inputs,
            init_rnn_state=(init_rnn_state.h_obs, init_rnn_state.c_obs) if init_rnn_state is not None else None
            )

        # ======================================================
        # pass CNN output + task embedding output throught goal generator
        # ======================================================
        goal, goal_history, (h_goal, c_goal) = self.goal_generator(
            obs_emb=image_embedding,
            task_emb=mission_embedding,
            init_goal_state=(init_rnn_state.h_goal, init_rnn_state.c_goal) if init_rnn_state is not None else None
            )
        variables['goal'] = goal

        variables['normalized_history'] = self.normalize_history
        if self.normalize_history:
            goal_history = F.normalize(goal_history + 1e-6, p=2, dim=-1)

        variables['goal_history'] = goal_history

        # Model should always leave B-dimension in rnn state: [N,B,H].
        # will reuse "RNN" state for sum/lstm goal trackers
        next_rnn_state = RnnState(h_obs=h_obs, c_obs=c_obs, h_goal=h_goal, c_goal=c_goal)


        # ======================================================
        # Compute Predictive Features using history of goals
        # + task
        # ======================================================
        goal_pred_input = torch.cat((goal, goal_history, mission_embedding), dim=-1)
        goal_predictions = self.goal_gvf(goal_pred_input)
        # TB x |A| x D
        goal_predictions = goal_predictions.view(T, B, self.output_size, self.head_size) 

        variables['goal_predictions'] = goal_predictions
        
        # ======================================================
        # get output of RL head
        # ======================================================
        if self.obs_in_state:
            state_action = torch.cat((goal_predictions, duplicate_vector(obs_mem_outputs, n=self.output_size, dim=2)), dim=-1)

        else:
            state_action = goal_predictions


        if all_variables:
            self.rl_head(state_action, mission_embedding, 
                final_fn=partial(restore_leading_dims, lead_dim=lead_dim, T=T, B=B),
                variables=variables)
            return variables
        else:
            rl_out = self.rl_head(state_action, mission_embedding)
            # Restore leading dimensions: [T,B], [B], or [], as input.
            rl_out = restore_leading_dims(rl_out, lead_dim, T, B)
            return list(rl_out) + [next_rnn_state]


class DqnGvfHead(torch.nn.Module):
    """docstring for DQNHead"""
    def __init__(self, input_size, head_size, output_size, task_dim, nonlinearity=torch.nn.ReLU, **kwargs):
        super(DqnGvfHead, self).__init__()
        self.head_size = head_size
        self.output_size = output_size
        self.successor_head = MlpModel(input_size, head_size, nonlinearity=nonlinearity, output_size=head_size)
        self.task_weights = nn.Linear(task_dim, head_size)

    def forward(self, state_action, task, final_fn=lambda x:x, variables=None):
        """
        """
        T, B, A = state_action.shape[:3]


        # TBA x H
        successor_features = self.successor_head(state_action.view(T*B*A, -1))
        # T X B X A X D
        successor_features = successor_features.view(T, B, A, self.head_size)


        # T X B X D
        weights = self.task_weights(task)
        # T X B X 1 X D
        weights = weights.unsqueeze(2)
        # T X B X D X 1 (for dot-product)
        weights = weights.transpose(-2, -1)

        # dot product
        q_values = torch.matmul(successor_features, weights).squeeze(-1)

        q_values = q_values.view(T*B, -1)
        q_values = final_fn(q_values)

        if variables is not None:
            variables['q'] = q_values
        else:
            return [q_values]
