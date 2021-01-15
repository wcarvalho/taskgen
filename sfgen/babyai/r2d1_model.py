import numpy as np
import torch
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from sfgen.babyai.modules import BabyAIConv, LanguageModel, BabyAIFiLMModulation, GatedModulation
from sfgen.babyai.task_gated_lstm import TaskGatedLSTM

RnnState = namedarraytuple("RnnState", ["h", "c"])




class BabyAIR2d1Model(torch.nn.Module):
    """2D convolutional neural network (for multiple video frames per
    observation) feeding into an LSTM and MLP output for Q-value outputs for
    the action set."""
    def __init__(
            self,
            image_shape,
            output_size,
            vision_model='babyai',
            lang_model='bigru',
            task_modulation='babyai',
            lstm_type='task_modulated',
            mission_shape=None,
            direction_shape=None,
            use_pixels=True, # whether using input pixels as input
            use_bow=False, # bag-of-words representation for vectors in symbolic input tensor
            endpool=True, # avoid pooling
            film_bias=True,
            batch_norm=False, 
            lstm_size=512,
            head_size=512,
            fc_size=512,
            text_embed_size=128,
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
        if mission_shape != None:
            self.word_rnn = LanguageModel(lang_model,
                input_dim=mission_shape[-1], 
                text_embed_size=text_embed_size,
                batch_first=True)

        else:
            self.word_rnn = None
            text_embed_size = 0



        # -----------------------
        # task-modulation
        # -----------------------
        if task_modulation == "babyai":
            self.task_modulation = BabyAIFiLMModulation(
                task_dim=text_embed_size,
                conv_feature_dims=self.conv.output_dims,
                fc_size=fc_size,
                film_kwargs=dict(
                    batchnorm=batch_norm,
                    onpolicy=False,
                    bias=film_bias,
                    )
                )
        elif task_modulation == "choplot":
            self.task_modulation = GatedModulation(
                task_dim=text_embed_size,
                conv_feature_dims=self.conv.output_dims,
                fc_size=fc_size,
                )
        elif task_modulation == "none":
            # no task modulation
            self.task_modulation = MlpModel(np.prod(self.conv.output_dims), fc_size, nonlinearity=torch.nn.ReLU())
        else:
            raise NotImplementedError(f"No support for '{task_modulation}'")





        # -----------------------
        # lstm
        # -----------------------
        lstm_input_size = 0
        lstm_input_size += self.task_modulation.output_size # image embedding
        lstm_input_size += output_size # action
        lstm_input_size += 1           # reward
        lstm_input_size += direction_embed_size

        if task_modulation == "none":
            # feed task into lstm
            lstm_input_size += text_embed_size


        self.lstm_type = lstm_type
        self.lstm_size = lstm_size
        if lstm_type == 'regular':
            self.lstm = torch.nn.LSTM(lstm_input_size, lstm_size)
        elif lstm_type == 'task_modulated':
            self.lstm = TaskGatedLSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_size,
                task_size=text_embed_size,
                )
        else:
            raise NotImplementedError


        # -----------------------
        # Q-network
        # -----------------------
        self.dueling = dueling
        head_input_size = lstm_size
        if task_modulation != "none":
            # task as input to Q-value, not LSTM
            head_input_size += text_embed_size

        if dueling:
            self.head = DuelingHeadModel(head_input_size, head_size, output_size)
        else:
            self.head = MlpModel(head_input_size, head_size, output_size=output_size)


    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        lstm_inputs = []


        # ======================================================
        # Process image
        # ======================================================
        # process image
        img = observation.image
        img = img.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)


        # ======================================================
        # Read mission + direction information in
        # ======================================================
        if 'mission' in observation and self.word_rnn:
            mission = observation.mission.long()
            mission_embedding = self.word_rnn(mission.view(T*B, mission.shape[-1])) # Fold if T dimension.
            # assert len(out.shape) == 3, "should always be (T*)B x N x D"
            if self.task_modulation == "none":
                lstm_inputs.append(mission_embedding.view(T, B, -1))

        if 'direction' in observation:
            direction = self.text_embedding(observation.direction.long())
            lstm_inputs.append(direction.view(T, B, -1))

        # ======================================================
        # Apply Conv + task modulation
        # ======================================================
        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        modulated_conv_out = self.task_modulation(conv_out, mission_embedding)


        # ======================================================
        # LSTM
        # ======================================================
        # -----------------------
        # get inputs
        # -----------------------
        lstm_inputs.extend([
            modulated_conv_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ])
        lstm_input = torch.cat(lstm_inputs, dim=2)

        # -----------------------
        # run through lstm
        # -----------------------
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        if self.lstm_type == 'regular':
            lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
            # T x B x D, (1 x B x D, 1 x B x D)
        elif self.lstm_type == 'task_modulated':
            lstm_out, (hn, cn) = self.lstm(input=lstm_input, state=init_rnn_state, task=mission_embedding.view(T, B, -1))
            # T x B x D, (1 x B x D, 1 x B x D)
        else:
            raise NotImplementedError()

        # if B != 1:
        #     import ipdb; ipdb.set_trace()
        # if B != 1 and T != 1:
        #     import ipdb; ipdb.set_trace()


        # ======================================================
        # Compute Q-values
        # ======================================================
        if self.task_modulation != "none":
            q_input = torch.cat((lstm_out.view(T * B, -1), mission_embedding.view(T * B, -1)), dim=-1)
        else:
            q_input = lstm_out.view(T * B, -1)
        q = self.head(q_input)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, next_rnn_state
