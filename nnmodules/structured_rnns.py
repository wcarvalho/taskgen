import torch
import torch.nn as nn
from rlpyt.utils.quick_args import save__init__args
from nnmodules.rl_lstm import RLLSTM

class ListStructuredRnn(nn.Module):
    """docstring for ListStructuredRnn"""
    def __init__(self, num, input_size, hidden_size, rnn_class='rllstm', **kwargs):
        super(ListStructuredRnn, self).__init__()
        save__init__args(locals())

        if rnn_class == 'rllstm':
            _rnn_class = RLLSTM
        elif rnn_class == 'lstm':
            _rnn_class = torch.nn.LSTM
        else:
            raise NotImplementedError(rnn)

        self.rnns = nn.ModuleList([_rnn_class(input_size=input_size, hidden_size=hidden_size, **kwargs) for _ in range(num)])

    def forward(self, x, init_state=None, done=None):
        """Assume 1st two dimensions are T=time, B=batch

        Args:
            x (TYPE): Description
            init_state (None, optional): Description
        """
        outs = []
        hs = []
        cs = []

        # ======================================================
        # break up prior state into pieces
        # ======================================================
        T, B = x.shape[:2]
        if init_state is not None:
            (init_h, init_c) = init_state
            init_h = init_h.view(1, B, self.num, self.hidden_size)
            init_c = init_c.view(1, B, self.num, self.hidden_size)

        # ======================================================
        # go through each 'head' 1 by 1
        # ======================================================
        for idx, rnn in enumerate(self.rnns):
            args = [x[:,:, idx]]

            if init_state is None:
                args.append(init_state)
            else:
                args.append((init_h[:,:,idx].contiguous(), init_c[:,:,idx].contiguous()))

            if self.rnn_class == 'rllstm':
                args.append(done)
            elif self.rnn_class == 'lstm':
                pass

            out, (h, c) = rnn(*args)

            outs.append(out)
            hs.append(h)
            cs.append(c)

        # ======================================================
        # stack full histories so keep heads
        # concatenate (h, c)
        # ======================================================
        outs = torch.stack(outs, dim=2)

        # turn into T x B x N*D
        hs = torch.cat(hs, dim=2)
        cs = torch.cat(cs, dim=2)


        return outs, (hs, cs)
