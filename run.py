import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import csv
import os

from datetime import datetime
from torch.autograd import Variable
from utils import *


LOGGER = logging.getLogger(__name__)


def run_nmt(model, loader, dataset, args, train=False):
    total_step = 0.0
    stats = {'loss': AverageMeter(),
             'bleu': AverageMeter()}

    for b_idx, batch in enumerate(loader):
        batch_size = batch.src[0].size(0)

        # Grad zero + mode change
        model.optimizer.zero_grad()
        if train: model.train()
        else: model.eval()

        # Get outputs
        outputs = model(*batch.src, *batch.trg)
        loss = model.get_loss(outputs, batch.trg[0][:,1:])
        stats['loss'].update(loss.item(), batch_size)

        # Optimize model
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.params, args.grad_max_norm)
            model.optimizer.step()
        
        # Print for print step or at last
        if b_idx % args.print_step == 0 or b_idx == (len(loader) - 1):
            _progress = (
                '{}/{} | Loss: {:.3f} | Total F1: {:.3f} | '.format(
                b_idx + 1, len(loader), stats['loss'].avg, 0)
            )
            LOGGER.info(_progress)

    return loss

