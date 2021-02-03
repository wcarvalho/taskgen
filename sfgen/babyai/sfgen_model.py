import numpy as np
import torch
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from sfgen.babyai.modules import initialize_parameters, ObservationLSTM
from sfgen.babyai.visual_goal_generator import VisualGoalGenerator
from sfgen.babyai.babyai_model import BabyAIModel, DQNHead, PPOHead

class SFGenModel(BabyAIModel):
    """
    """
    def __init__(
        self,
        output_size,
        pre_mod_layer=False, # whether to apply layer to goal/task before obtaining cnn weights
        mod_function='sigmoid',
        mod_compression='maxpool',
        goal_tracking='sum',
        lstm_size=512,
        head_size=512,
        obs_fc_size=512,
        dueling=False,
        rlhead='dqn',
        AuxTaskCls=None,
        **kwargs
        ):
        """
        """
        super(SFGenModel, self).__init__(
            **kwargs
        )
        assert dueling == False, "Successor doesn't suppport dueling currently"

        self.observation_memory = ObservationLSTM(
            conv_feature_dims=self.conv.output_dims,
            lstm_size=lstm_size,
            fc_size=obs_fc_size,
            # action dim + reward + direction
            extra_input_dim=output_size+1+self.direction_embed_size,
        )

        goal_dim = self.conv.output_dims[0] # number of channels
        self.goal_generator = VisualGoalGenerator(
            conv_feature_dims=self.conv.output_dims,
            task_dim=self.text_embed_size,
            goal_dim=goal_dim, # number of channels
            pre_mod_layer=pre_mod_layer,
            mod_function=mod_function,
            mod_compression=mod_compression,
            goal_tracking=goal_tracking,
        )

        input_size = 2*goal_dim + lstm_size

        if rlhead == 'dqn':
            self.rl_head = DQNHead(
                input_size=lstm_size + self.text_embed_size,
                head_size=head_size,
                output_size=output_size,
                dueling=dueling)

        elif rlhead == 'successor_dqn':
            self.rl_head = DQNSuccessorHead(
                input_size=lstm_size,
                head_size=head_size,
                output_size=output_size,
                task_dim=self.text_embed_size)

        elif rlhead == 'ppo':
            self.rl_head = PPOHead(
                input_size=lstm_size + self.text_embed_size, 
                output_size=output_size,
                hidden_size=head_size)
            self.apply(initialize_parameters)
        else:
            raise RuntimeError(f"RL Algorithm '{rlhead}' unsupported")

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""


        lead_dim, T, B, img_shape = infer_leading_dims(observation.image, 3)

        image_embedding, mission_embedding, direction_embeding = self.process_observation(observation)

        # ======================================================
        # pass through Observation LSTM
        # ======================================================
        non_obs_inputs = [e for e in [direction_embeding] if e is not None]
        non_obs_inputs.extend([prev_action, prev_reward])
        obs_mem_outputs, (h_obs, c_obs) = self.observation_memory(
            obs_emb=image_embedding,
            init_lstm_inputs=non_obs_inputs,
            init_rnn_state=(init_rnn_state.h_obs, init_rnn_state.c_obs) if init_rnn_state is not None else None
            )

        # ======================================================
        # pass throught goal generator
        # ======================================================
        goal, goal_mem_outputs, (h_goal, c_goal) = self.goal_generator(
            obs_emb=image_embedding,
            task_emb=mission_embedding,
            init_goal_state=(init_rnn_state.h_goal, init_rnn_state.c_goal) if init_rnn_state is not None else None
            )


        # Model should always leave B-dimension in rnn state: [N,B,H].
        # will reuse "RNN" state for sum/lstm goal trackers
        next_rnn_state = RnnState(h_obs=h_obs, c_obs=c_obs, h_goal=h_goal, c_goal=c_goal)

        # ======================================================
        # get output of RL head
        # ======================================================
        state_vars = [goal, goal_mem_outputs, obs_mem_outputs]
        state_vars = [r.view(T, B, -1) for r in state_vars]


        mission_embedding = mission_embedding.view(T, B, -1)


        rl_out = self.rl_head(state_vars, mission_embedding)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        rl_out = restore_leading_dims(rl_out, lead_dim, T, B)

        return list(rl_out) + [next_rnn_state]


class DQNSuccessorHead(torch.nn.Module):
    """docstring for DQNHead"""
    def __init__(self, input_size, head_size, output_size, task_dim, **kwargs):
        super(DQNSuccessorHead, self).__init__()
        self.head_size = head_size
        self.output_size = output_size
        self.successor_head = MlpModel(input_size, head_size, output_size=head_size*output_size)
        self.task_weights = nn.Linear(task_dim, head_size)

    def forward(self, state_variables, task):
        """
        """
        state_variables.append(task)
        state = torch.cat(state_variables, dim=-1)
        T, B = state.shape[:2]
        
        # TB x A x H
        successor_features = self.successor_head(state.view(T*B, -1)).view(T*B, self.output_size, self.head_size)
        weights = self.task_weights(task)

        q = torch.matmul(successor_features, weights)

        return [q]
