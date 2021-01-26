"""
Adapted from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

"""
# from collections import namedtuple
from typing import List, Tuple

import torch
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter


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
        # mu = torch.mm(task, self.weight_mu.t())
        # if self.training:
        #     logvar = torch.mm(task, self.weight_sigma.t())
        #     sigma = logvar.mul(0.5).exp()
        #     # dist = torch.normal(mu=mu, std=sigma)
        #     eps = torch.empty_like(sigma).normal_()
        #     if self.init_hidden:
        #         raise NotImplemented("Always initialize hidden as 0 vector")
        #     else:
        #         cell_init = eps.mul(sigma).add_(mu)
        # else:
        #     if self.init_hidden:
        #         raise NotImplemented("Always initialize hidden as 0 vector")
        #     else:
        #         cell_init = mu

        B = task.shape[0]
        zeros = torch.zeros(B, self.cell.hidden_size,
            dtype=task.dtype, device=task.device)
        return (zeros, zeros)

    def forward(self, input, state, task):
        """Squeeze/unsqueeze data so follows convention of pytorch's LSTM class
        
        if state is None, it's constructed.
        if task has only length=1 along time-dimension, same value is used at every time-step.
        Args:
            input (TYPE): T x B x D
            state (TYPE): 1 x B x D or None
            task (TYPE): T x B x D or 1 x B x D

        """

        if state is None:
            assert task.shape[0] == 1, "require 1 time-step for task when initializing state"
            state = self.init_state(task.squeeze(0))
        else:
            hx, cx = state
            assert len(hx.shape) == 3 and hx.shape[0] == 1, "only support having 1 x B x D state tensors"
            state = (hx.squeeze(0), cx.squeeze(0))

        if task.shape[0] == 1:
            outputs, (hn, cn) = self.process(input, state, task.squeeze(0))
        elif task.shape[:2] == input.shape[:2]:
            outputs, (hn, cn) = self.process(input, state, task, timewise_task_input=True)
        else:
            raise RuntimeError(f"Task shape: {task.shape}. Input shape: {input.shape}")


        hn, cn = (hn.unsqueeze(0), cn.unsqueeze(0))
        return outputs, (hn, cn)



    @jit.script_method
    def process(self, input, state, task, timewise_task_input=False):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            if timewise_task_input:
                out, state = self.cell(inputs[i], state, task[i])
            else:
                out, state = self.cell(inputs[i], state, task)
            outputs += [out]

        return torch.stack(outputs), state

def main():
    lstm = TaskGatedLSTM(512, 512, 512)


if __name__ == '__main__':
    main()