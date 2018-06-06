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
DATA_PATH = os.path.join(DATA_DIR, 'wmt14.pkl')
RESULTS_DIR = './results'
LOG_DIR = os.path.join(RESULTS_DIR, 'log')
MODEL_NAME = 'test.mdl'


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

argparser.add_argument('--data-path', type=str, default=DATA_PATH,
                       help='Dataset path')
argparser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                       help='Directory for model results')
argparser.add_argument('--model-name', type=str, default=MODEL_NAME,
                       help='Model name for saving/loading')
argparser.add_argument('--print-step', type=float, default=100,
                       help='Display steps')
argparser.add_argument('--validation-step', type=float, default=1,
                       help='Number of random search validation')
argparser.add_argument('--train', type='bool', default=True,
                       help='Enable training')
argparser.add_argument('--valid', type='bool', default=True,
                       help='Enable validation')
argparser.add_argument('--test', type='bool', default=True,
                       help='Enable testing')
argparser.add_argument('--resume', type='bool', default=False,
                       help='Resume saved model')
argparser.add_argument('--debug', type='bool', default=False,
                       help='Run as debug mode')

# Train config
argparser.add_argument('--batch-size', type=int, default=32)
argparser.add_argument('--epoch', type=int, default=40)
argparser.add_argument('--learning-rate', type=float, default=5e-4)
argparser.add_argument('--weight-decay', type=float, default=0)
argparser.add_argument('--grad-max-norm', type=int, default=10)
argparser.add_argument('--grad-clip', type=int, default=10)

# Model config
argparser.add_argument('--binary', type='bool', default=False)
argparser.add_argument('--hidden-dim', type=int, default=576)
argparser.add_argument('--lstm-layer', type=int, default=1)
argparser.add_argument('--lstm-dr', type=float, default=0.0)
argparser.add_argument('--char-dr', type=float, default=0.0)
argparser.add_argument('--bi-lstm', type='bool', default=True)
argparser.add_argument('--linear-dr', type=float, default=0.0)
argparser.add_argument('--char-embed-dim', type=int, default=20)
argparser.add_argument('--seed', type=int, default=3)

args = argparser.parse_args()


def run_experiment(model, dataset, run_fn, args, cell_line=None):

    # Get dataloaders
    train_loader, valid_loader, test_loader = dataset.get_dataloader(
        batch_size=args.batch_size, s_idx=args.s_idx) 

    # Save and load model during experiments
    if args.train:
        if args.resume:
            model.load_checkpoint(args.results_dir, args.model_name)

        best = 0.0
        converge_cnt = 0
        decay_cnt = 0

        for ep in range(args.epoch):
            LOGGER.info('Training Epoch %d' % (ep+1))
            run_fn(model, train_loader, dataset, args, train=True)

            if args.valid:
                LOGGER.info('Validation')
                curr = run_fn(model, valid_loader, dataset, args, train=False)

                # If best model, save
                if not args.resume and curr > best:
                    best = curr
                    model.save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()},
                        args.results_dir, args.model_name)
                    converge_cnt = 0
                else:
                    converge_cnt += 1

                # If converged, decay lr
                if converge_cnt >= 3:
                    for param_group in model.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    converge_cnt = 0
                    decay_cnt += 1

                    # If decayed more than 3 times, stop
                    if decay_cnt > 3:
                        LOGGER.info('Early stopping applied')
                        break
                    else:
                        LOGGER.info('Decaying {}: learning rate {:.4f}'.format(
                            deacy_cnt, model.optimizer.param_groups[0]['lr']))

    
    if args.test:
        LOGGER.info('Performance Test on Valid & Test Set')
        if args.train or args.resume:
            model.load_checkpoint(args.results_dir, args.model_name)
        run_fn(model, valid_loader, dataset, args, metric, train=False)
        run_fn(model, test_loader, dataset, args, metric, train=False)


def get_dataset(path):
    return pickle.load(open(path, 'rb'))


def get_run_fn(args):
    return run_nmt


def get_model(args, dataset):
    model = Seq2SeqAttModel(input_dim=dataset.input_dim,
                            g_dropout=args.g_dropout).cuda()
                            
    return model


def init_logging(args):
    import uuid
    import time
    log_id = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    # For logfile writing
    logfile = logging.FileHandler(
        os.path.join(LOG_DIR, log_id + '.txt'), 'w')
    logfile.setFormatter(fmt)
    LOGGER.addHandler(logfile)


def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def init_parameters(args, model_name, model_idx):
    args.model_name = '{}-{}'.format(model_name, model_idx)
    '''
    args.learning_rate = np.random.uniform(1e-4, 2e-3)
    args.batch_size = 2 ** np.random.randint(4, 7)
    args.grad_max_norm = 5 * np.random.randint(1, 5)
    args.hidden_dim = 64 * np.random.randint(1, 10)
    '''


def main():
    # Initialize logging
    init_logging(args)
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
