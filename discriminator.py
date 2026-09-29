import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

class Block(nn.Module):
    def __init__(self, in_dim, med_dim, out_dim, s_filters):
        super().__init__()

        self.block = nn.Sequential(
            SN(nn.Conv1d(in_dim, med_dim, s_filters[0], padding=(s_filters[0]-1)//2)),
            nn.LeakyReLU(),
            SN(nn.Conv1d(med_dim, out_dim, s_filters[1], padding=(s_filters[1]-1)//2)),
            nn.LeakyReLU())
        
        self.conv1x1 = nn.Conv1d(in_dim, out_dim, 1)
    
    def forward(self, x):
        x = self.conv1x1(x) + self.block(x)
        x = F.leaky_relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, seq_len, num_embs, vocab_size, emb_dim, med_dims, out_dims, s_filters, hid_dim):
        super(Discriminator, self).__init__()
        
        self.embs = nn.ModuleList([nn.Linear(vocab_size, emb_dim) for i in range(num_embs)])

        self.disc = nn.Sequential(
            Block(emb_dim, med_dims[0], out_dims[0], s_filters),
            Block(out_dims[0], med_dims[1], out_dims[1], s_filters),
            Block(out_dims[1], med_dims[2], out_dims[2], s_filters)
        )

        self.flat = nn.Flatten()

        d = seq_len * out_dims[-1]
        
        self.lin1 = SN(nn.Linear(d, hid_dim))
        self.lin2 = SN(nn.Linear(hid_dim, 1))

        self.out = nn.Sigmoid()

    def forward(self, x):
        xs = [emb(x) for emb in self.embs]
        
        outs = []
        for xi in xs:
            o = self.disc(xi.permute(0, 2, 1))
            o = self.flat(o)
            o = F.leaky_relu(self.lin1(o))
            o = self.out(self.lin2(o))
            outs.append(o.squeeze())
        
        return outs