import torch
import torch.nn as nn
import numpy as np
from rlpyt.models.mlp import MlpModel

from nnmodules.modules import BabyAIFiLMModulation, GatedModulation
from nnmodules.task_gated_lstm import TaskGatedLSTM


class ModulatedMemory(nn.Module):
    """docstring for ModulatedMemory"""
    def __init__(self,
        action_dim,
        conv_feature_dims,
        task_modulation='film',
        lstm_type='task_gated',
        lstm_size=512,
        fc_size=512,
        text_embed_size=128,
        nonmodulated_input_size=0,
        film_bias=True,
        film_residual=True,
        film_pool=False,
        batch_norm=False,
        ):
        super(ModulatedMemory, self).__init__()
        self.lstm_type = lstm_type

        # ======================================================
        # task-based observation modulation
        # ======================================================
        if task_modulation == "film":
            self.task_modulation = BabyAIFiLMModulation(
                task_dim=text_embed_size,
                conv_feature_dims=conv_feature_dims,
                fc_size=fc_size,
                residual=film_residual,
                pool=film_pool,
                film_kwargs=dict(
                    batchnorm=batch_norm,
                    onpolicy=False,
                    bias=film_bias,
                    )
                )
            modulated_input_size = self.task_modulation.output_size
        elif task_modulation == "chaplot":
            self.task_modulation = GatedModulation(
                task_dim=text_embed_size,
                conv_feature_dims=conv_feature_dims,
                fc_size=fc_size,
                )
            modulated_input_size = self.task_modulation.output_size
        elif task_modulation == "none":
            flat_dims = np.prod(conv_feature_dims)
            # no task modulation
            if fc_size:
                self.task_modulation = MlpModel(flat_dims, fc_size, nonlinearity=nn.ReLU())
            else:
                self.task_modulation = lambda obs, task : obs
                modulated_input_size = flat_dims
        else:
            raise NotImplementedError(f"No support for '{task_modulation}'")



        # ======================================================
        # task-modulated memory
        # ======================================================


        lstm_input_size = 0
        lstm_input_size += modulated_input_size # modulated image embedding
        lstm_input_size += action_dim # action
        lstm_input_size += 1           # reward
        lstm_input_size += nonmodulated_input_size # e.g. direction
        if lstm_type == 'regular':
            self.lstm = nn.LSTM(lstm_input_size, lstm_size)
        elif lstm_type == 'task_gated':
            self.lstm = TaskGatedLSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_size,
                task_size=text_embed_size,
                )
        else:
            raise NotImplementedError

    def forward(self, obs_emb, task_emb, init_lstm_inputs=[], init_rnn_state=None):
        """
        - modulate image observation
        - collection [modulated input, prev reward, prev action] and pass through lstm

        Args:
            obs_emb (TYPE): T X B X ...
            task_emb (TYPE): Description
            init_lstm_inputs (list, optional): Description
            init_rnn_state (None, optional): Description
        
        Returns:
            TYPE: Description
        
        Raises:
            NotImplementedError: Description
        """
        # init_lstm_inputs = list(init_lstm_inputs) # copy list

        T, B = obs_emb.shape[:2]
        # -----------------------
        # modulate obs
        # -----------------------
        modulated_obs_emb = self.task_modulation(obs_emb.flatten(0,1), task_emb.flatten(0,1))

        # -----------------------
        # run through lstm
        # -----------------------
        lstm_inputs = init_lstm_inputs + [modulated_obs_emb]
        lstm_inputs = [e.view((T, B, -1)) for e in lstm_inputs]
        lstm_input = torch.cat(lstm_inputs, dim=2)

        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)

        # outputs should be
        # T x B x D, (1 x B x D, 1 x B x D)
        if self.lstm_type == 'regular':
            lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        elif self.lstm_type == 'task_gated':
            lstm_out, (hn, cn) = self.lstm(input=lstm_input, state=init_rnn_state, task=task_emb.view(T, B, -1))
        else:
            raise NotImplementedError()

        return lstm_out, (hn, cn)


class DualBodyModulatedMemory(nn.Module):
    """
    Copies architecture from ModulatedMemory, except there's a second branch which processes all information and not task-modulated information
    """
    def __init__(self,
        action_dim,
        lstm_size=512,
        dual_body=True,
        task_modulation='babyai',
        lstm_type='task_gated',
        **kwargs):
        super(DualBodyModulatedMemory, self).__init__()

        individual_size = lstm_size//2 if dual_body else lstm_size
        self.modulated_mem = ModulatedMemory(
            action_dim=action_dim,
            task_modulation=task_modulation,
            lstm_type=lstm_type,
            lstm_size=individual_size,
             **kwargs)

        self.dual_body = dual_body
        if dual_body:
            self.reg_mem = ModulatedMemory(
                action_dim=action_dim,
                task_modulation='none',
                lstm_type='regular',
                lstm_size=individual_size,
                 **kwargs)


    def forward(self, init_rnn_state=None, **kwargs):

        if init_rnn_state is not None:
            mod_state = (init_rnn_state.hmod, init_rnn_state.cmod)
            reg_state = (init_rnn_state.hreg, init_rnn_state.creg)

        else:
            mod_state = None
            reg_state = None

        outm, (hm, cm) = self.modulated_mem(init_rnn_state=mod_state, **kwargs)
        if self.dual_body:
            outr, (hr, cr) =  self.reg_mem(init_rnn_state=reg_state, **kwargs)
        else:
            outr = torch.zeros_like(hm, device=outm.device, dtype=outm.dtype)
            hr = torch.zeros_like(hm, device=hm.device, dtype=hm.dtype)
            cr = torch.zeros_like(hm, device=cm.device, dtype=cm.dtype)

        return outm, (hm, cm), outr, (hr, cr)

