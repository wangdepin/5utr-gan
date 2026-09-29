import numpy as np

DIC = {'A': [1., 0., 0., 0.],
       'T': [0., 1., 0., 0.],
       'C': [0., 0., 1., 0.],
       'G': [0., 0., 0., 1.],
       'N': [0., 0., 0., 0.]}

def one_hot_encoding(path, max_len):
    utr = []
    with open(path, 'r') as fin:
        for line in fin:
            utr_seq = line.strip()

            if len(utr_seq) < max_len:
                utr_seq = 'N'*(max_len-len(utr_seq))+utr_seq

            elif len(utr_seq) > max_len:
                utr_seq = utr_seq[-max_len:]

            sutr = []
            for base in utr_seq[::-1]:
                sutr.append(DIC[base])
            
            utr.append(sutr)
    
    utr = np.array(utr) # shape: [N, L, C]

    return utr
