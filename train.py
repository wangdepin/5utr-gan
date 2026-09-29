import os
import random
import time
import math

import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GenDataset, DisDataset, PAD_INDEX

from torch.utils.tensorboard import SummaryWriter

from generator import Generator
from discriminator import Discriminator
from encoding import one_hot_encoding

timestr = time.strftime("%Y%m%d%H%M", time.localtime())
print('Timestamp: %s' % timestr)


# ================== Parameter Definition =================
parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--gpu', type=int, default=0, help='The GPU used to train the model')
parser.add_argument('--log_dir', type=str, default='results/%s/log' % timestr, help='Log dir of the trained model')
parser.add_argument('--out_dir', type=str, default='results/%s' % timestr, help='Directory of generated sequences')
parser.add_argument('--backup', type=bool, default=True, help='Whether to backup scripts')
parser.add_argument('--tau', type=float, default=1, help='Temperature parameter of gumbel_softmax')
parser.add_argument('--max_len', type=int, default=50, help='Sequence length the model is trained on')
parser.add_argument('--pre_epochs', type=int, default=100, help='Generator MLE pretraining epochs')
parser.add_argument('--total_batch', type=int, default=500, help='Adversarial training steps')
parser.add_argument('--d_steps', type=int, default=5, help='Discriminator updates per generator update')
parser.add_argument('--d_pretrain_steps', type=int, default=30, help='Discriminator minibatches per pretraining epoch')
parser.add_argument('--sample_every', type=int, default=25, help='Write generated sequences every N steps')
parser.add_argument('--objective', choices=['nll', 'bce_gumbel'], default='nll',
                    help="Generator pretraining objective. 'nll' is the likelihood; "
                         "'bce_gumbel' reproduces the original BCE-on-a-Gumbel-sample "
                         "objective for ablation.")
opt = parser.parse_args()

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

if opt.backup:
    os.system("zip -r %s/codes.zip *.py" % opt.out_dir)

writer = SummaryWriter(opt.log_dir)

# Basic Training Paramters
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = False

BATCH_SIZE = 64
TOTAL_BATCH = opt.total_batch
GENERATED_NUM = 5000

POS_REAL_FILE = 'data/train.pos'
NEG_REAL_FILE = 'data/train.neg'

VOCAB_SIZE = 4
PRE_EPOCH_NUM = opt.pre_epochs

DE_VOCABS = {0:'A', 1:'T', 2:'C', 3:'G'}

if torch.cuda.is_available():
    device = 'cuda:%d' % opt.gpu
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print('Device: %s' % device)

max_len = opt.max_len
alpha = 0.01 # label smoothing factor

# Genrator Parameters
dims = [64, 128, 128, 128, 256, 256, 256]
tau = opt.tau

# Discriminator Parameters
num_embs = 32
d_emb_dim = 64
med_dims = [64, 64, 128]#, 128]
out_dims = [64, 128, 128]#, 256]
s_filters = [3, 5]
d_hid_dim = 128

writer.add_scalar('hyperparams/batch_size', BATCH_SIZE)
writer.add_scalar('hyperparams/batch_nums', TOTAL_BATCH)
writer.add_scalar('hyperparams/max_len', max_len)
writer.add_scalar('hyperparams/pre_epochs', PRE_EPOCH_NUM)
writer.add_scalar('hyperparams/tau', tau)

def label_smoothing(label, alpha):
    return label * (1 - alpha) + alpha / 3

def generate_samples(generator, batch_size, generated_num, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    fout = open(output_file, 'a')
    for _ in range(int(generated_num / batch_size)):
        samples = generator.sample(batch_size).data.cpu().numpy()
        
        for s in samples:
            string = ''.join([DE_VOCABS[np.argmax(c)] for c in s])
            fout.write('%s\n' % string[::-1])

    fout.close()

def train_generator(generator, dataloader, criterion, optimizer):
    total_loss = 0.
    nb_batches = len(dataloader)

    for batch, (X, Y) in enumerate(tqdm(dataloader, ncols=60)):
        X = X.to(device)
        Y = Y.to(device)

        logp = generator(X)

        if opt.objective == 'nll':
            loss = criterion(logp.reshape(-1, VOCAB_SIZE), Y.reshape(-1))
        else:
            # Original objective: binary cross-entropy against a *sampled* Gumbel-Softmax
            # relaxation rather than the log-probabilities. Kept as an ablation.
            pred = F.gumbel_softmax(logp, tau=generator.tau)
            onehot = F.one_hot(Y.clamp(min=0), VOCAB_SIZE).float()
            onehot[Y == PAD_INDEX] = 0.
            loss = criterion(pred, onehot)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / nb_batches

def train_discriminator(discriminator, generator, dataloader, criterion, optimizer, max_steps=None):
    '''Update the discriminator on real/shuffled pairs and on real/generated pairs.

    `max_steps` caps the number of minibatches so the adversarial loop can run a
    fixed, small number of discriminator updates per generator update instead of
    a full pass over the data.
    '''
    total_loss = 0.
    nb_batches = len(dataloader) if max_steps is None else min(max_steps, len(dataloader))

    for batch, (pX, nX) in enumerate(tqdm(dataloader, ncols=60, total=nb_batches)):
        if batch >= nb_batches:
            break

        rX = torch.cat([pX, nX], dim=0).to(device)
        rY = torch.FloatTensor([1]*pX.size(0)+[0]*nX.size(0)).to(device)
        rY = label_smoothing(rY, alpha)
        rpreds = discriminator(rX)

        fX = generator.sample(rX.size(0)).detach() # fake X

        pX = pX.to(device)
        gX = torch.cat([pX, fX], dim=0).to(device)
        gY = torch.FloatTensor([1]*pX.size(0)+[0]*fX.size(0)).to(device)
        gY = label_smoothing(gY, alpha)
        gpreds = discriminator(gX)
        
        losses = []
        for rpred in rpreds:
            loss = criterion(rpred, rY)
            losses.append(loss)
        
        for gpred in gpreds:
            loss = criterion(gpred, gY)
            losses.append(loss)
        
        avg_loss = sum(losses) / (num_embs * 2)

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        total_loss += avg_loss.item()
    
    return total_loss / nb_batches

def main():
    # Define Networks
    generator = Generator(max_len, VOCAB_SIZE, dims, device)
    discriminator = Discriminator(max_len, num_embs, VOCAB_SIZE, d_emb_dim, med_dims, out_dims, s_filters, d_hid_dim)
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Load data 
    pos_real_data = one_hot_encoding(POS_REAL_FILE, max_len)
    neg_real_data = one_hot_encoding(NEG_REAL_FILE, max_len)

    # Construct dataloader for Generator
    pos_real_dataset = GenDataset(pos_real_data)
    pos_real_dataloader = DataLoader(pos_real_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Construct dataloader for Classifier
    dis_data = [pos_real_data, neg_real_data]
    dis_dataset = DisDataset(dis_data)
    dis_dataloader = DataLoader(dis_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Pretrain Generator using MLE
    # NLLLoss on the log-probabilities is the actual likelihood objective; the
    # padding positions flagged by GenDataset are skipped.
    gen_criterion = (nn.NLLLoss(ignore_index=PAD_INDEX) if opt.objective == 'nll'
                     else nn.BCELoss())
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)

    print('\nPretrain Generator with MLE ...')
    for epoch in range(1, PRE_EPOCH_NUM+1):
        if epoch % 3 == 0:
            generator.tau = generator.tau * 0.955

        pos_loss = train_generator(generator, pos_real_dataloader, gen_criterion, gen_optimizer)
        print(f'Epoch: {epoch}; generate loss: {pos_loss:.5e}')
        writer.add_scalar('loss/G_pretrain_loss', pos_loss, epoch)
    
    # Generate samples
    GEN_FILE = '%s/gen_5utr_pretrain' % opt.out_dir
    generate_samples(generator, BATCH_SIZE, GENERATED_NUM, GEN_FILE)
    
    # Pretrain Discriminator
    dis_criterion = nn.BCELoss()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=2e-3)

    # Each discriminator step draws a fresh batch of generated sequences, which
    # is the dominant cost, so the pass is capped rather than run to completion.
    print('\nPretrain Discriminator ...')
    for epoch in range(5):
        dis_loss = train_discriminator(discriminator, generator, dis_dataloader,
                                       dis_criterion, dis_optimizer, max_steps=opt.d_pretrain_steps)
        print(f'Epoch: {epoch+1}; total disc loss: {dis_loss:.5e}')
        writer.add_scalar('loss/D_pretrain_loss', dis_loss, epoch+1)

    # Adversarial Training
    print('\n#####################################################')
    print('Start Adeversatial Training...')
    
    gan_loss = nn.BCELoss()
    
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        if (total_batch+1) % 20 == 0:
            generator.tau = generator.tau * 0.955

        samples = generator.sample(BATCH_SIZE*2)
        label = torch.FloatTensor([1.]*samples.size(0)).to(device)
        label = label_smoothing(label, alpha)
        preds = discriminator(samples) 
        
        losses = []
        for pred in preds:
            loss = gan_loss(pred, label)
            losses.append(loss)
        
        avg_loss = sum(losses) / num_embs
            
        gen_optimizer.zero_grad()
        avg_loss.backward()
        gen_optimizer.step()

        gloss = avg_loss.item()

        dis_loss = train_discriminator(discriminator, generator, dis_dataloader,
                                       dis_criterion, dis_optimizer, max_steps=opt.d_steps)

        print(f'Batch: {total_batch+1}; GAN-GEN Loss: {gloss:.5e}; total disc loss: {dis_loss:.5e}')
        writer.add_scalar('loss/G_loss', gloss, total_batch+1)
        writer.add_scalar('loss/D_loss', dis_loss, total_batch+1)

        # Generate samples
        if (total_batch+1) % opt.sample_every == 0 or total_batch+1 == TOTAL_BATCH:
            GEN_FILE = '%s/gen_5utr_batch%d' % (opt.out_dir, total_batch+1)
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, GEN_FILE)

    writer.close()

if __name__ == '__main__':
    main()
 