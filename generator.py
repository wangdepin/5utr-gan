import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k, s, d, p, drop=0.2):
        '''
        k: kernel size
        s: stride
        d: dilation
        p: padding
        '''
        super(TemporalBlock, self).__init__()
        self.block = nn.Sequential(
            wn(nn.Conv1d(in_dim, out_dim, k, stride=s, padding=p, dilation=d)),
            Chomp(p),
            nn.ReLU(),
            nn.Dropout(drop),
            wn(nn.Conv1d(out_dim, out_dim, k, stride=s, padding=p, dilation=d)),
            Chomp(p),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.downsample = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def forward(self, x):
        x1 = self.block(x)
        x2 = x if self.downsample is None else self.downsample(x)
        x = F.relu(x1 + x2)

        return x

class TemporalConvNet(nn.Module):
    def __init__(self, dims, k=2, drop=0.2):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        for i in range(len(dims)-1):
            d = 2 ** i
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers += [TemporalBlock(in_dim, out_dim, k, s=1, d=d, p=(k-1)*d, drop=drop)]

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.blocks(x)
        return x

class Generator(nn.Module):
    def __init__(self, seql, vocab_size, dims, device, k=2, drop=0.2, tau=1.0):
        super().__init__()
        self.seql = seql
        self.vocab_size = vocab_size
        self.device = device
        self.tau = tau

        self.emb = nn.Linear(vocab_size, dims[0])
        self.tcn = TemporalConvNet(dims, k=k, drop=drop)
        self.lin1 = nn.Linear(dims[-1], 64)
        self.lin2 = nn.Linear(64, vocab_size)
        self.logp = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
        Log-probabilities for every position in a single pass.

        The convolutions are causal, so position i only sees inputs up to i and
        one pass yields the whole sequence of next-base distributions.

        x: (N, L, C) one-hot input, shifted right by one position
        returns: (N, L, C) log-probabilities
        '''
        x = self.emb(x)
        x = self.tcn(x.permute(0, 2, 1))

        x = F.relu(self.lin1(x.permute(0, 2, 1)))

        return self.logp(self.lin2(x))

    def sample(self, batch_size):
        '''
        Autoregressively draw one-hot sequences of length seql.

        Gumbel-Softmax with hard=True is a straight-through estimator: the
        forward pass emits a true one-hot base, the backward pass uses the soft
        relaxation, so the discriminator's gradient reaches the generator.
        '''
        x = torch.zeros((batch_size, 1, self.vocab_size)).to(self.device)

        for _ in range(self.seql):
            logp = self.forward(x)[:, -1:, :]
            x_ = F.gumbel_softmax(logp, tau=self.tau, hard=True)
            x = torch.cat([x, x_], dim=1)

        return x[:, 1:, :]