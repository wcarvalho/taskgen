"""
Adapted from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

"""
# from collections import namedtuple
from typing import List, Tuple

import torch
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter


class RLLSTM(jit.ScriptModule):
    def __init__(self, lstm_cell=torch.nn.LSTMCell, **kwargs):
        super(RLLSTM, self).__init__()
        self.cell = lstm_cell(**kwargs)

    def init_state(self, x):
        if x is None:
            raise NotImplementedError("Need example")
        T, B = x.shape[:2]
        zeros = torch.zeros(B, self.cell.hidden_size,
            dtype=x.dtype, device=x.device)
        return (zeros, zeros)

    def forward(self, inputs, state, done):
        """Squeeze/unsqueeze data so follows convention of nnmodules's LSTM class

        if state is None, it's constructed.

        Args:
            inputs (TYPE): T x B x D
            state (TYPE): 1 x B x D or None
            done (TYPE): Description
        
        Returns:
            TYPE: Description
        
        """

        if state is None:
            state = self.init_state(inputs)
        else:
            hx, cx = state
            assert len(hx.shape) == 3 and hx.shape[0] == 1, "only support having 1 x B x D state tensors"
            state = (hx.squeeze(0), cx.squeeze(0))

        outputs, (hn, cn) = self.process(inputs, state, done)

        # assert len(hn.shape) == 2
        hn, cn = (hn.unsqueeze(0), cn.unsqueeze(0))
        return outputs, (hn, cn)

    @jit.script_method
    def step(self, obs, state, done):
        # type: (Tensor, Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx = self.cell(obs, state)
        if done is not None and done.sum():
            h0, c0 = self.init_state(obs.unsqueeze(0))
            float_done = done.float().unsqueeze(1)

            hx = h0*float_done + (1-float_done)*hx
            cx = c0*float_done + (1-float_done)*cx

        return hx, cx

    @jit.script_method
    def process(self, inputs, state, done):
        # type: (Tensor, Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = inputs.unbind(0)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            done_i = done[i] if done is not None else None
            hx, cx = self.step(inputs[i], state, done_i)
            state = (hx, cx)
            outputs += [hx]

        return torch.stack(outputs), state

def main():
    lstm = RLLSTM(512, 512, 512)


if __name__ == '__main__':
    main()