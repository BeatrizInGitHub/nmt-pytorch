import os
import sys
import logging
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from functools import partial
from torch.autograd import Variable

from dataset import WMTDataset
from run import *
from model import Seq2SeqAttModel
from utils import *


LOGGER = logging.getLogger()

DATA_DIR = './data'
DATA_PATH = os.path.join(DATA_DIR, 'wmt(small).pkl')
RESULTS_DIR = './results'
LOG_DIR = os.path.join(RESULTS_DIR, 'log')
MODEL_NAME = 'test_mdl'


# Create dirs
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


# Run settings
argparser = argparse.ArgumentParser()
argparser.register('type', 'bool', str2bool)

argparser.add_argument('--data-path', type=str, default=DATA_PATH)
argparser.add_argument('--results-dir', type=str, default=RESULTS_DIR)
argparser.add_argument('--model-name', type=str, default=MODEL_NAME)
argparser.add_argument('--print-step', type=float, default=30)
argparser.add_argument('--validation-step', type=float, default=1)
argparser.add_argument('--train', type='bool', default=True)
argparser.add_argument('--valid', type='bool', default=True)
argparser.add_argument('--test', type='bool', default=True)
argparser.add_argument('--resume', type='bool', default=False)
argparser.add_argument('--debug', type='bool', default=False)

# Train config
argparser.add_argument('--batch-size', type=int, default=32)
argparser.add_argument('--epoch', type=int, default=5)
argparser.add_argument('--learning-rate', type=float, default=1e-3)
argparser.add_argument('--grad-max-norm', type=int, default=1)

# Model config
argparser.add_argument('--rnn-dim', type=int, default=1000)
argparser.add_argument('--rnn-layer', type=int, default=1)
argparser.add_argument('--rnn-dropout', type=float, default=0.0)
argparser.add_argument('--bi-rnn', type='bool', default=True)
argparser.add_argument('--word-embed-dim', type=int, default=620)
argparser.add_argument('--align-dim', type=int, default=1000)
argparser.add_argument('--maxout-dim', type=int, default=500)
argparser.add_argument('--linear-dropout', type=float, default=0.0)
argparser.add_argument('--seed', type=int, default=3)

args = argparser.parse_args()


def run_experiment(model, dataset, run_fn, args, cell_line=None):

    # Save and load model during experiments
    if args.train:
        if args.resume:
            model.load_checkpoint(args.results_dir, args.model_name)

        best = 0.0
        for ep in range(args.epoch):
            LOGGER.info('Training Epoch %d' % (ep+1))
            run_fn(model, dataset.train_iter, dataset, args, train=True)

            if args.valid:
                LOGGER.info('Validation')
                curr = run_fn(model, dataset.valid_iter, dataset, args, train=False)

                # If best model, save
                if not args.resume and curr > best:
                    best = curr
                    model.save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()},
                        args.results_dir, args.model_name)

    if args.test:
        LOGGER.info('Performance Test on Valid & Test Set')
        if args.train or args.resume:
            model.load_checkpoint(args.results_dir, args.model_name)
        run_fn(model, dataset.valid_iter, dataset, args, train=False)
        run_fn(model, dataset.test_iter, dataset, args, train=False)


def get_dataset(path):
    dataset = WMTDataset()
    dataset.load(path)
    return dataset


def get_run_fn(args):
    return run_nmt


def get_model(args, dataset):
    model = Seq2SeqAttModel(
        rnn_dim=args.rnn_dim,
        rnn_layer=args.rnn_layer,
        rnn_dropout=args.rnn_dropout,
        bi_rnn=args.bi_rnn,
        pad_idx=dataset.src.vocab.stoi['<pad>'],
        word_embed_dim=args.word_embed_dim,
        align_dim=args.align_dim,
        maxout_dim=args.maxout_dim,
        src_vocab_size=len(dataset.src.vocab),
        trg_vocab_size=len(dataset.trg.vocab),
        linear_dropout=args.linear_dropout,
        learning_rate=args.learning_rate,
    ).cuda()
                            
    return model


def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def init_parameters(args, model_name, model_idx):
    args.model_name = '{}_{}'.format(model_name, model_idx)


def main():
    # Initialize logging
    init_logging(LOGGER, LOG_DIR)
    LOGGER.info('COMMAND: {}'.format(' '.join(sys.argv)))

    # Get datset, run function, model name
    dataset = get_dataset(args.data_path)
    run_fn = get_run_fn(args)
    model_name = args.model_name

    # Random search validation
    for model_idx in range(args.validation_step):
        LOGGER.info('Validation step {}'.format(model_idx+1))

        # Initialize seed, and randomize params if needed
        init_seed(args.seed)
        init_parameters(args, model_name, model_idx)
        LOGGER.info(args)

        # Get model object
        model = get_model(args, dataset)

        # Run experiment
        run_experiment(model, dataset, run_fn, args)


if __name__ == '__main__':
    main()
