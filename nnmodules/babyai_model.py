from functools import partial
import torch
import torch.nn.functional as F
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.quick_args import save__init__args

from sfgen.babyai.modules import BabyAIConv, LanguageModel, initialize_parameters
from pytorch.modulation_architectures import DualBodyModulatedMemory

RnnState = namedarraytuple("RnnState", ["hmod", "cmod", "hreg", "creg"])
# Encoding = namedarraytuple("Encoding", ["direction", "mission", "image"])


class BabyAIModel(torch.nn.Module):
    """
    """
    def __init__(
            self,
            image_shape,
            mission_shape=None,
            direction_shape=None,
            vision_model='babyai',
            lang_model='bigru',
            use_pixels=True, # whether using input pixels as input
            use_bow=False, # bag-of-words representation for vectors in symbolic input tensor
            batch_norm=False, 
            text_embed_size=128,
            text_output_size=0,
            direction_embed_size=32,
            nonlinearity='ReLU',
            # endpool=True, # avoid pooling
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            **kwargs,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        save__init__args(locals())

        self.nonlinearity_fn = getattr(torch.nn, nonlinearity)
        # -----------------------
        # embedding for direction
        # -----------------------
        if direction_shape != None:
            self.direction_embedding = torch.nn.Embedding(
                num_embeddings=4,
                embedding_dim=direction_embed_size,
            )
        else:
            self.direction_embedding = None
            self.direction_embed_size = 0


        # -----------------------
        # conv for image
        # -----------------------
        if vision_model.lower() == 'atari':
            assert use_pixels, "atari CNN only works with pixel observations"
            raise NotImplementedError("Need to figure out output h, w, c")
            self.conv = Conv2dHeadModel(
                image_shape=image_shape,
                channels=channels or [32, 64, 64],
                kernel_sizes=kernel_sizes or [8, 4, 3],
                strides=strides or [4, 2, 1],
                paddings=paddings or [0, 1, 1],
                use_maxpool=use_maxpool,
                hidden_sizes=None,  # conv features as output
                nonlinearity=self.nonlinearity_fn
            )
        elif vision_model.lower() == 'babyai':
            self.conv = BabyAIConv(
                image_shape=image_shape,
                use_bow=use_bow,
                use_pixels=use_pixels,
                endpool=not use_maxpool,
                batch_norm=batch_norm,
                nonlinearity=self.nonlinearity_fn
            )
        else:
            raise NotImplemented(f"Don't know how to support '{vision_model}' vision model")

        # -----------------------
        # embedding for instruction
        # -----------------------
        self.text_embed_size = text_embed_size
        if mission_shape != None:
            self.word_rnn = LanguageModel(lang_model,
                input_dim=mission_shape[-1], 
                text_embed_size=text_embed_size,
                batch_first=True,
                output_dim=text_output_size,
                )

        else:
            self.word_rnn = None
            self.text_embed_size = 0


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

        image_embedding = self.conv(img.view(T * B, *img_shape)).view(T, B, *self.conv.output_dims)

        # ======================================================
        # Read mission + direction information in
        # ======================================================
        mission_embedding = None
        if 'mission' in observation and self.word_rnn:
            mission = observation.mission.long()
            mdim = mission.shape[-1]
            mission_embedding = self.word_rnn(mission.view(T*B, mdim)).view(T, B, -1) # Fold if T dimension.

        direction_embedding = None
        if 'direction' in observation:
            direction_embedding = self.text_embedding(observation.direction.long())
            raise RuntimeError("Never checked dimensions work out")

        return image_embedding, mission_embedding, direction_embedding

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        raise NotImplementedError

class BabyAIRLModel(BabyAIModel):
    """
    """
    def __init__(
        self,
        output_size,
        dual_body=True,
        task_modulation='film',
        lstm_type='task_gated',
        film_bias=True,
        lstm_size=512,
        head_size=512,
        fc_size=512,
        dueling=False,
        rlhead='dqn',
        film_batch_norm=False,
        film_residual=True,
        film_pool=False,
        intrustion_policy_input=False,
        **kwargs
        ):
        """
        """
        super(BabyAIRLModel, self).__init__(**kwargs)
        self.dual_body = dual_body
        self.intrustion_policy_input = intrustion_policy_input
        self.memory = DualBodyModulatedMemory(
            action_dim=output_size,
            conv_feature_dims=self.conv.output_dims,
            task_modulation=task_modulation,
            lstm_type=lstm_type,
            film_bias=film_bias,
            lstm_size=lstm_size,
            fc_size=fc_size,
            dual_body=dual_body,
            # direction only thing not modulated (IF given)
            nonmodulated_input_size=self.direction_embed_size, 
            # if batchnorm is on, this is on. if batchnorm is off, fil setting has to be invidually on. no way to batchnorm conv features but not film.
            batch_norm=film_batch_norm or self.batch_norm, 
            film_residual=film_residual,
            film_pool=film_pool,
            )

        if intrustion_policy_input:
            input_size = lstm_size + self.text_embed_size
        else:
            raise RuntimeError("Always set intrustion_policy_input=True")
            input_size = lstm_size
        
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

    def forward(self, observation, prev_action, prev_reward, init_rnn_state, all_variables=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        variables=dict()
        lead_dim, T, B, img_shape = infer_leading_dims(observation.image, 3)

        image_embedding, mission_embedding, direction_embedding = self.process_observation(observation)

        if all_variables:
            variables['image_embedding'] = image_embedding
            variables['mission_embedding'] = mission_embedding
            variables['direction_embedding'] = direction_embedding

        # ======================================================
        # pass through LSTM
        # ======================================================
        non_mod_inputs = [e for e in [direction_embedding] if e is not None]
        non_mod_inputs.extend([prev_action, prev_reward])

        outm, (hm, cm), outr, (hr, cr) = self.memory(
            obs_emb=image_embedding,
            task_emb=mission_embedding,
            init_lstm_inputs=non_mod_inputs,
            init_rnn_state=init_rnn_state,
            )
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(hmod=hm, cmod=cm, hreg=hr, creg=cr)

        if all_variables:
            variables['next_rnn_state'] = next_rnn_state
        # ======================================================
        # get output of RL head
        # ======================================================
        mission_embedding = mission_embedding.view(T, B, -1)
        # combine LSTM outputs with mission embedding
        if self.dual_body:
            rl_input = [outm, outr]
        else:
            rl_input = [outm]

        # give mission embedding to policy as well?
        # if self.intrustion_policy_input:
        #     rl_input.append(mission_embedding)

        # cat copies. can avoid copy operation with this
        # if len(rl_input) > 1:
        #     rl_input = torch.cat(rl_input, dim=-1)
        # else:
        #     rl_input = rl_input[0]


        if all_variables:
            self.rl_head(rl_input, mission_embedding, 
                final_fn=partial(restore_leading_dims, lead_dim=lead_dim, T=T, B=B),
                variables=variables)
            return variables
        else:
            rl_out = self.rl_head(rl_input, mission_embedding)
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
