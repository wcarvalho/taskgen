"""
Most of file adapted from:
https://github.com/mila-iqia/babyai/blob/master/babyai/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dModel



# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# ======================================================
# Instruction Processing
# ======================================================
class LanguageModel(nn.Module):
    """Directly borrowed from babyAI codebase:

    Changes:
    - removed support for attention-based gru

    """
    def __init__(self, lang_model, input_dim, text_embed_size, batch_first=True):
        super(LanguageModel, self).__init__()
        self.lang_model = lang_model
        self.batch_first = batch_first
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=text_embed_size,
            )
        if lang_model in ['gru', 'bigru']:
            gru_dim = text_embed_size
            if lang_model in ['bigru']:
                gru_dim //= 2

            self.gru = torch.nn.GRU(
                text_embed_size, gru_dim, batch_first=batch_first,
                bidirectional=(lang_model in ['bigru']))

        else:
            raise NotImplementedError

    def forward(self, instruction):
        B = instruction.shape[0]
        lengths = (instruction != 0).sum(1).long()

        embedding = self.word_embedding(instruction)
        if self.lang_model == 'gru':
            out, _ = self.gru(embedding)
            return out[np.arange(B),lengths-1]

        elif self.lang_model == 'bigru':
            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instruction.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                # inputs = self.word_embedding(instr)
                inputs = embedding[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=self.batch_first)

                outputs, final_states = self.gru(inputs)
            else:
                instruction = instruction[:, 0:lengths[0]]
                outputs, final_states = self.gru(self.word_embedding(instruction))
                iperm_idx = None

            # 2 x B x D/2 --> B x 2 x D/2
            final_states = final_states.transpose(0, 1).contiguous()
            # B x 2 x D/2 --> B x D
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                # outputs, _ = pad_packed_sequence(outputs, batch_first=self.batch_first)
                # outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return final_states

# ======================================================
# Input Tensor Processing (e.g. image or symbolic input)
# ======================================================

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self,
        task_dim,
        out_features,
        in_channels,
        mid_channels,
        batchnorm=False,
        onpolicy=False,
        bias=True,
        ):
        """Addded:
        - onpolicy
        - bias
        
        Args:
            task_dim (TYPE): Description
            out_features (TYPE): Description
            in_channels (TYPE): Description
            mid_channels (TYPE): Description
            batchnorm (bool, optional): Description
            onpolicy (bool, optional): Description
            bias (bool, optional): Description
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels,
            kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(out_features)
        else:
            self.bn1 = lambda x : x
            self.bn2 = lambda x : x

        self.weight = nn.Linear(task_dim, out_features)

        self.bias = None
        if bias:
            self.bias = nn.Linear(task_dim, out_features)

        if onpolicy:
            self.apply(initialize_parameters)

    def forward(self, conv, task):
        conv = F.relu(self.bn1(self.conv1(conv)))
        conv = self.conv2(conv)
        weight = self.weight(task).unsqueeze(2).unsqueeze(3)

        if self.bias:
            bias = self.bias(task).unsqueeze(2).unsqueeze(3)
        else:
            bias = 0
        out = conv * weight + bias
        return F.relu(self.bn2(out))

class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim, onpolicy=False):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       if onpolicy:
           self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)

class BabyAIConv(nn.Module):
    """
    CNN used by BabyAI paper. Taken directly from their code (below).

    Modifications made to remove things like batch_norm and pooling which aren't typically used with reinforcement learning.

    Diagrams of models:
        - https://arxiv.org/pdf/2007.12770.pdf
    """
    def __init__(self, use_bow=False, use_pixels=True, endpool=True, batch_norm=False, image_shape=None):
        super(BabyAIConv, self).__init__()
        # only apply bag-of-word rep if not using pixels
        use_bow = use_bow if not use_pixels else False

        self.model = nn.Sequential(*[
            *([ImageBOWEmbedding(image_shape, 128)] if use_bow else []),
            *([nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=(8, 8),
                stride=8, padding=0)] if use_pixels else []),
            nn.Conv2d(
                in_channels=128 if use_bow or use_pixels else 3, out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            *(nn.BatchNorm2d(128) if batch_norm else []),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            *(nn.BatchNorm2d(128) if batch_norm else []),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])

        x = torch.zeros((1, *image_shape))
        y = self.model(x)

        self._output_size = y.shape[1]
        self._output_dims = y.shape[1:]

    def forward(self, x): 
        return self.model(x)
        # return y.flatten(-3,-1)

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_dims(self):
        return self._output_dims


# ======================================================
# Modulation Architectures
# ======================================================
class BabyAIFiLMModulation(nn.Module):
    """docstring for BabyAIFiLMModulation"""
    def __init__(self, task_dim, conv_feature_dims, num_module=2, residual=True, fc_size=None, film_kwargs={}):
        super(BabyAIFiLMModulation, self).__init__()
        self.num_module = num_module
        self.residual = residual

        self.controllers = []

        channels, height , width = conv_feature_dims

        for ni in range(num_module):
            mod = FiLM(
                in_channels=channels,
                mid_channels=128,
                out_features=128 if ni < num_module-1 else channels,
                task_dim=task_dim,
                **film_kwargs)
            self.controllers.append(mod)
            # so .to(device) works on these
            self.add_module('FiLM_' + str(ni), mod)


        if fc_size:
            self.final_layer = MlpModel(
                input_size=channels*height*width,
                hidden_sizes=[fc_size],
                output_size=None, 
                nonlinearity=torch.nn.ReLU)
            self._output_size = fc_size
        else:
            self.final_layer = lambda x:x
            self._output_size = channels*height*width


    def forward(self, conv, task):
        for controller in self.controllers:
            out = controller(conv, task)
            if self.residual:
                out += conv
            conv = out

        out = self.final_layer(conv.view(conv.shape[0], -1))
        return out

    @property
    def output_size(self):
        return self._output_size


# Inspired by Gated-Attention architectures from https://arxiv.org/pdf/1706.07230.pdf
class GatedModulation(nn.Module):
    """docstring for GatedModulation"""
    def __init__(self, task_dim, conv_feature_dims, fc_size=None):
        super(GatedModulation, self).__init__()
        self.task_dim = task_dim
        self.conv_feature_dims = conv_feature_dims

        channels, height , width = conv_feature_dims

        self.weight = nn.Linear(task_dim, channels)

        if fc_size:
            self.final_layer = MlpModel(
                input_size=channels*height*width,
                hidden_sizes=[fc_size],
                output_size=None, 
                nonlinearity=torch.nn.ReLU)
            self._output_size = fc_size
        else:
            self.final_layer = lambda x:x
            self._output_size = channels*height*width

    def forward(self, conv, task):
        weight = self.weight(task).unsqueeze(2).unsqueeze(3)
        out = conv * weight

        out = self.final_layer(out.view(out.shape[0], -1))

        return out

    @property
    def output_size(self):
        return self._output_size



