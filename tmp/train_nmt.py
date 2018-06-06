import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from nltk.translate.bleu_score import sentence_bleu

from dataset import *
from model import SEQ2SEQ

import argparse


parser = argparse.ArgumentParser(
    description='Neural Machine Translation with WMT14 en => fr'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')

args = parser.parse_args()


mb_size = 32
h_dim = 200
lr = 1e-3
lr_decay_every = 1000000
n_iter = 10000000
log_interval = 1000

dataset = WMT_Dataset()

model = SEQ2SEQ(
    dataset.source_n_vocab, dataset.target_n_vocab,
    h_dim, dataset.emb_dim, p_word_dropout=0.3,
    gpu=args.gpu
)

def model_params(_model):
    print('model parameters: ', end='')
    params = list()
    total_size = 0

    def multiply_iter(p_list):
        out = 1
        for _p in p_list:
            out *= _p
        return out

    for p in _model.parameters():
        if p.requires_grad:
            params.append(p)
            total_size += multiply_iter(p.size())
    print('%s' % '{:,}'.format(total_size))
    return params


print(model)
model_params(model)


def main():
    trainer = optim.Adam(model.s2s_params, lr=lr)

    for it in range(n_iter):
        inputs, targets = dataset.next_batch(args.gpu)
        # print(dataset.source_idxs2sentence(inputs[:,0]))
        # print(dataset.target_idxs2sentence(targets[:,0]))

        loss = model.forward(inputs)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.s2s_params, 5)
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            sample_translation = model.translate_sentences(inputs)
            score = [sentence_bleu([dataset.target_idxs2sentence(targets[:,k], True)],
                                   dataset.target_idxs2sentence(sample_translation[k], True),
                                   weights=[1,0,0,0])
                     for k in range(targets.size(1))]
            score = sum(score)/len(score)

            print('Iter-{}; Loss: {:.4f}; BLEU: {:.4f}'.format(it, loss.data[0], score))

            print('Source: "{}"'.format(dataset.source_idxs2sentence(
                                        inputs[:,0])))
            print('Target: "{}"'.format(dataset.target_idxs2sentence(
                                        targets[:,0])))
            print('Result: "{}"'.format(dataset.target_idxs2sentence(
                                        sample_translation[0])))
            print()


def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/vae.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if args.save:
            save_model()

        exit(0)

    if args.save:
        save_model()
