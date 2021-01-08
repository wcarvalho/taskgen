
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


RnnState = namedarraytuple("RnnState", ["h", "c"])


class BabyAIR2d1Model(torch.nn.Module):
    """2D convolutional neural network (for multiple video frames per
    observation) feeding into an LSTM and MLP output for Q-value outputs for
    the action set."""
    def __init__(
            self,
            image_shape,
            output_size,
            lang_model='bigru',
            mission_shape=None,
            direction_shape=None,
            fc_size=512,  # Between conv and lstm.
            lstm_size=512,
            head_size=512,
            text_embed_size=256,
            direction_embed_size=32,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling

        # -----------------------
        # conv for image
        # -----------------------
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_size,  # ReLU applied here (Steven).
        )

        # -----------------------
        # embedding for text
        # -----------------------
        if mission_shape != None:
            self.word_embedding = torch.nn.Embedding(
                num_embeddings=mission_shape[-1],
                embedding_dim=text_embed_size,
                )
            self.word_rnn = self.build_language_model(lang_model, text_embed_size)

        else:
            self.word_embedding = None
            text_embed_size = 0

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
            direction_embed_size = 0

        self.lstm_input_size = 0
        self.lstm_input_size += self.conv.output_size # image
        self.lstm_input_size += output_size # action
        self.lstm_input_size += 1           # reward
        self.lstm_input_size += direction_embed_size
        self.lstm_input_size += text_embed_size

        self.lstm = torch.nn.LSTM(self.lstm_input_size, lstm_size)


        if dueling:
            self.head = DuelingHeadModel(lstm_size, head_size, output_size)
        else:
            self.head = MlpModel(lstm_size, head_size, output_size=output_size)

    def build_language_model(self, lang_model, text_embed_size):
        """Directly borrowed from babyAI codebase:
        https://github.com/mila-iqia/babyai/blob/master/babyai/model.py
        """
        if lang_model in ['gru', 'bigru', 'attgru']:
            if lang_model in ['gru', 'bigru', 'attgru']:
                gru_dim = text_embed_size
                if lang_model in ['bigru', 'attgru']:
                    gru_dim //= 2
                return torch.nn.GRU(
                    text_embed_size, gru_dim, batch_first=True,
                    bidirectional=(lang_model in ['bigru', 'attgru']))

        else:
            raise NotImplementedError
    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        lstm_inputs = []


        # process image
        img = observation.image
        img = img.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        # only have word embedding if have more than 1 mission
        if 'mission' in observation and self.word_embedding:
            mission = self.word_embedding(observation.mission.long())
            if len(mission.shape) == 2:
                # N X D --> 1 X N X D
                out, _ = self.word_rnn(mission.unsqueeze(0))
                # final time-step
                lstm_inputs.append(out[:, -1].view(T, B, -1))
            elif len(mission.shape) == 3:
                # B X N X D
                out, _ = self.word_rnn(mission)
                # out = out
                lstm_inputs.append(out[:, -1].view(T, B, -1))
            else:
                # T X B X N X D
                out, _ = self.word_rnn(mission.flatten(0,1))
                # out = out
                lstm_inputs.append(out[:, -1].view(T, B, -1))

        if 'direction' in observation:
            direction = self.text_embedding(observation.direction.long())
            lstm_inputs.append(direction.view(T, B, -1))




        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.


        lstm_inputs.extend([
            conv_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ])

        lstm_input = torch.cat(lstm_inputs, dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

        q = self.head(lstm_out.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, next_rnn_state
