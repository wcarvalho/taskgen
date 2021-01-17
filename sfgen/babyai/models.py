import numpy as np
import torch
import torch.nn.functional as F
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from sfgen.babyai.modules import BabyAIConv, LanguageModel
from sfgen.babyai.modulation_architectures import ModulatedMemory, DualBodyModulatedMemory



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
            endpool=True, # avoid pooling
            batch_norm=False, 
            text_embed_size=128,
            direction_embed_size=32,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()


        # -----------------------
        # embedding for direction
        # -----------------------
        self.direction_embed_size = direction_embed_size
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
            )
        elif vision_model.lower() == 'babyai':
            self.conv = BabyAIConv(
                image_shape=image_shape,
                use_bow=use_bow,
                use_pixels=use_pixels,
                endpool=endpool,
                batch_norm=batch_norm
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
                batch_first=True)

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
        rlalgorithm='dqn',
        **kwargs
        ):
        """
        """
        super(BabyAIRLModel, self).__init__(**kwargs)
        self.dual_body = dual_body
        self.memory = DualBodyModulatedMemory(
            action_dim=output_size,
            conv_feature_dims=self.conv.output_dims,
            task_modulation=task_modulation,
            lstm_type=lstm_type,
            film_bias=film_bias,
            lstm_size=lstm_size,
            fc_size=fc_size,
            dual_body=dual_body,
            nonmodulated_input_size=self.direction_embed_size, # direction only thing not modulated (IF given)
            )

        input_size = lstm_size + self.text_embed_size
        if rlalgorithm == 'dqn':
            self.rl_head = DQNHead(input_size)
        elif rlalgorithm == 'ppo':
            self.rl_head = PPOHead(input_size, output_size)
        else:
            raise RuntimeError(f"RL Algorithm '{rlalgorithm}' unsupported")

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""


        lead_dim, T, B, img_shape = infer_leading_dims(observation.image, 3)

        image_embedding, mission_embedding, direction_embeding = self.process_observation(observation)

        # ======================================================
        # pass through LSTM
        # ======================================================
        non_mod_inputs = [e for e in [direction_embeding] if e is not None]
        non_mod_inputs.extend([prev_action, prev_reward])

        outm, (hm, cm), outr, (hr, cr) = self.memory(
            obs_emb=image_embedding,
            task_emb=mission_embedding,
            init_lstm_inputs=non_mod_inputs,
            init_rnn_state=init_rnn_state,
            )
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(hmod=hm, cmod=cm, hreg=hr, creg=cr)

        # ======================================================
        # get output of RL head
        # ======================================================
        # combine LSTM outputs with mission embedding
        if self.dual_body:
            rl_input = [outm, outr]
        else:
            rl_input = [outm]
        rl_input.append(mission_embedding.view(T, B, -1))
        rl_input = torch.cat(rl_input, dim=-1)

        rl_out = self.rl_head(rl_input)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        rl_out = restore_leading_dims(rl_out, lead_dim, T, B)

        return list(rl_out) + [next_rnn_state]

class PPOHead(torch.nn.Module):
    """
    """
    def __init__(self, input_size, output_size):
        super(PPOHead, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.pi = torch.nn.Linear(input_size, output_size)
        self.value = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        """
        """
        T, B = x.shape[:2]
        pi = F.softmax(self.pi(x.view(T * B, -1)), dim=-1)
        v = self.value(x.view(T * B, -1)).squeeze(-1)

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

    def forward(self, x):
        """
        """
        T, B = x.shape[:2]
        q = self.head(x.view(T * B, -1))

        return [q]
