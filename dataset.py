import numpy as np
import torch
from torch.utils.data import Dataset

PAD_INDEX = -100 # nn.NLLLoss ignore_index

class GenDataset(Dataset):
    '''Teacher-forcing pairs for generator pretraining.

    The input is the sequence shifted right by one position, so predicting
    target[i] may only use bases 0..i-1. Positions holding the all-zero padding
    vector carry PAD_INDEX and are excluded from the loss: the model should not
    be trained to emit a base where the source UTR had already ended.
    '''

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        utr = self.raw_data[idx]

        in_utr = np.concatenate((np.zeros((1, utr.shape[1])), utr[:-1, :]), axis=0)

        target = utr.argmax(axis=-1)
        target[utr.sum(axis=-1) == 0] = PAD_INDEX

        in_utr = torch.tensor(in_utr, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.long)

        return in_utr, target

class DisDataset(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data[0])

    def __getitem__(self, idx):
        utr = self.raw_data[0][idx]
        target = self.raw_data[1][idx]

        utr = torch.tensor(utr, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)

        return utr, target