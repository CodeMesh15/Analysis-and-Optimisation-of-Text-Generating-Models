# model/rec2_model.py
import torch
import torch.nn as nn

class Rec2Network(nn.Module):
    def __init__(self, n_vocab, r1_units, n_pos, r2_units):
        super(Rec2Network, self).__init__()
        self.embed_x = nn.Embedding(n_vocab, r1_units)
        self.r1 = nn.RNN(r1_units, r1_units, batch_first=True)
        self.embed_y = nn.Embedding(n_pos, r2_units)
        self.r2 = nn.RNN(r1_units + r2_units, r2_units, batch_first=True)
        self.output = nn.Linear(r2_units, n_pos)

    def forward(self, x, y):
        ex = self.embed_x(x)
        _, hx = self.r1(ex)

        ey = self.embed_y(y)
        h = torch.cat([ey, hx.repeat(ey.size(1), 1, 1).transpose(0, 1)], dim=2)

        h2, _ = self.r2(h)
        return self.output(h2)
