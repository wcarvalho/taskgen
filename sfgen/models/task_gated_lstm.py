"""
Adapted from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

"""
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit

# from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
# import numbers




class TaskGatedLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, task_size):
        super(TaskGatedLSTMCell, self).__init__()
        # -----------------------
        # original LSTM components
        # -----------------------
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

        # -----------------------
        # new gating componentsc
        # -----------------------
        self.task_size = task_size
        self.weight_th = Parameter(torch.randn(hidden_size, task_size))

    @jit.script_method
    def forward(self, input, state, task):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        modulation = torch.mm(task, self.weight_th.t())

        ingate = torch.sigmoid(ingate * modulation)
        forgetgate = torch.sigmoid(forgetgate * modulation)
        outgate = torch.sigmoid(outgate * modulation)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class TaskGatedLSTM(jit.ScriptModule):
    def __init__(self, *args, **kwargs):
        super(TaskGatedLSTM, self).__init__()
        self.cell = TaskGatedLSTMCell(*args, **kwargs)

    @jit.script_method
    def forward(self, input, state, task):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)

        if state is None:
            B = input.shape[0]
            D = self.cell.hidden_size
            zeros = torch.zeros(1, B, self.cell.hidden_size,
                                dtype=input.dtype, device=input.device)
            state = (zeros, zeros)


        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, task)
            outputs += [out]
        return torch.stack(outputs), state

def main():
    lstm = TaskGatedLSTM(512, 512)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()