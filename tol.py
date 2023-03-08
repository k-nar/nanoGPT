import torch
from torch import tensor, nn
from torch.nn import functional as F


class KeyTree(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.key = nn.Parameter(tensor(16, 382, dim))

    def forward(self, query):
        addresses = torch.zeros((16, 8), dtype=torch.int)
        for i in range(8): #(0:2),
            keys = self.key[:, 0:2]

            # query: (batch, dim)
            # key: (1022, dim)
            # out: (batch, 1022)
            out = torch.einsum('bd, cd -> bc', query, self.key[i])
            addresses.append(out)
        # query: (batch, dim)
        # key: (16, 1022, dim)
        # out: (batch, 16, 1022)
        return torch.einsum('bd, bcd -> bc', query, self.key)


