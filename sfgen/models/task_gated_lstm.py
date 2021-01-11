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
        self.weight_task_input = Parameter(torch.randn(hidden_size, task_size))
        self.weight_task_output = Parameter(torch.randn(hidden_size, task_size))
        self.weight_task_forget = Parameter(torch.randn(hidden_size, task_size))

    @jit.script_method
    def forward(self, input, state, task):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        input_modulation = torch.mm(task, self.weight_task_input.t())
        forget_modulation = torch.mm(task, self.weight_task_forget.t())
        output_modulation = torch.mm(task, self.weight_task_output.t())

        ingate = torch.sigmoid(ingate * input_modulation)
        forgetgate = torch.sigmoid(forgetgate * forget_modulation)
        outgate = torch.sigmoid(outgate * output_modulation)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class TaskGatedLSTM(jit.ScriptModule):
    def __init__(self, init_hidden=False, *args, **kwargs):
        super(TaskGatedLSTM, self).__init__()
        self.cell = TaskGatedLSTMCell(*args, **kwargs)

        self.init_hidden = init_hidden
        self.chunks = 2 if init_hidden else 1
        hidden_size = self.chunks*self.cell.hidden_size
        self.weight_sigma = Parameter(torch.randn(hidden_size, self.cell.task_size))
        self.weight_mu = Parameter(torch.randn(hidden_size, self.cell.task_size))


    def init_state(self, task):
        mu = torch.mm(task, self.weight_mu.t())
        if self.training:
            logvar = torch.mm(task, self.weight_sigma.t())
            sigma = logvar.mul(0.5).exp()
            # dist = torch.normal(mu=mu, std=sigma)
            eps = torch.empty_like(sigma).normal_()
            if self.init_hidden:
                raise NotImplemented("Always initialize hidden as 0 vector")
            else:
                cell_init = eps.mul(sigma).add_(mu)
        else:
            if self.init_hidden:
                raise NotImplemented("Always initialize hidden as 0 vector")
            else:
                cell_init = mu

        B = task.shape[0]
        zeros = torch.zeros(B, self.cell.hidden_size,
            dtype=task.dtype, device=task.device)
        return (zeros, cell_init)


    @jit.script_method
    def forward(self, input, state, task):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)

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