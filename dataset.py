import numpy as np
import copy
import pickle
import string
import os
import random
import torch
import spacy
import re
import sys
import logging
import dill
import torch
import pandas

from utils import init_logging
from torchtext import data, datasets

LOGGER = logging.getLogger()


class WMTDataset(object):
    def __init__(self):
        LOGGER.info('Loading field for source (en), target (fr)')
        self.src = data.Field(tokenize=data.get_tokenizer('spacy'), 
                              init_token='<start>',
                              eos_token='<eos>',
                              include_lengths=True,
                              batch_first=True,
                              lower=True)
        self.trg = data.Field(tokenize=data.get_tokenizer('spacy'), 
                              init_token='<start>',
                              eos_token='<eos>',
                              include_lengths=True,
                              batch_first=True,
                              lower=True)

    def preprocess(self, dirs, save_path=None):
        LOGGER.info('Preprocessing WMT data')
        paths = {
            'train': set(),
            'valid': set(),
            'test': set()
        }
        results = {
            'train': [],
            'valid': [],
            'test': [],
        }

        # Find files within the directories
        for subdir, _, files, in os.walk(dirs['train']):
            for file_name in sorted(files):
                file_path = os.path.join(subdir, file_name)
                file_path = file_path[:-3]
                paths['train'].update([file_path])

        for subdir, _, files, in os.walk(dirs['valid']):
            for file_name in sorted(files):
                file_path = os.path.join(subdir, file_name)
                file_path = file_path[:-3]
                paths['valid'].update([file_path])

        for subdir, _, files, in os.walk(dirs['test']):
            for file_name in sorted(files):
                file_path = os.path.join(subdir, file_name)
                file_path = file_path[:-3]
                paths['test'].update([file_path])

        # Set filter
        f = lambda ex: len(ex.src) <= 99999 and len(ex.trg) <= 99999

        # Preprocess datasets
        for train_path in sorted(paths['train']):
            # if 'nc9' not in train_path:
            #     continue
            LOGGER.info('Train data: {}'.format(train_path))
            train = datasets.TranslationDataset(
                exts=('.en', '.fr'), fields=(self.src, self.trg),
                path=train_path,
                filter_pred=f
            )
            results['train'] += vars(train)['examples']

        for valid_path in sorted(paths['valid']):
            LOGGER.info('Valid data: {}'.format(valid_path))
            valid = datasets.TranslationDataset(
                exts=('.en', '.fr'), fields=(self.src, self.trg),
                path=valid_path,
                filter_pred=f
            )
            results['valid'] += vars(valid)['examples']

        for test_path in sorted(paths['test']):
            LOGGER.info('Test data: {}'.format(test_path))
            test = datasets.TranslationDataset(
                exts=('.en', '.fr'), fields=(self.src, self.trg),
                path=test_path,
                filter_pred=f
            )
            results['test'] += vars(test)['examples']
        
        # Save
        if save_path is not None:
            torch.save(results, save_path)

        self.dataset = results

    def load(self, load_path=None, train_bs=80, valid_bs=128, test_bs=128):
        if load_path is not None:
            self.dataset = torch.load(load_path)

        LOGGER.info('Train: {}, Valid: {}, Test: {}'.format(
            len(self.dataset['train']),
            len(self.dataset['valid']),
            len(self.dataset['test'])))

        # Reload datasets
        fields = [('src', self.src), ('trg', self.trg)]
        train = data.Dataset(fields=fields, examples=self.dataset['train'])
        valid = data.Dataset(fields=fields, examples=self.dataset['valid'])
        test = data.Dataset(fields=fields, examples=self.dataset['test'])

        # Build vocabulary
        LOGGER.info('Building vocabulary')
        self.src.build_vocab(train, max_size=30000)
        self.trg.build_vocab(train, max_size=30000)

        # Ready for iterators
        LOGGER.info('Setting iterators')
        self.train_iter = data.BucketIterator(
            train, batch_size=train_bs,
            shuffle=True, repeat=False, device=torch.device('cuda'),
            sort_within_batch=True,
            sort_key=lambda x: len(x.src)
        )
        self.valid_iter = data.BucketIterator(
            valid, batch_size=valid_bs,
            shuffle=False, repeat=False, device=torch.device('cuda'),
            sort_within_batch=True,
            sort_key=lambda x: len(x.src)
        )
        self.test_iter = data.BucketIterator(
            test, batch_size=test_bs,
            shuffle=False, repeat=False, device=torch.device('cuda'),
            sort_within_batch=True,
            sort_key=lambda x: len(x.src)
        )

    def src_idx2word(self, idxs, split=False):
        eos_idx = torch.argmax(idxs == self.src.vocab.stoi['<eos>'])
        if split:
            return [self.src.vocab.itos[i] for i in idxs[1:eos_idx]]
        else:
            return ' '.join([self.src.vocab.itos[i] for i in idxs[1:eos_idx]])

    def trg_idx2word(self, idxs, split=False):
        eos_idx = torch.argmax(idxs == self.trg.vocab.stoi['<eos>'])
        if eos_idx == 0:
            eos_idx = torch.LongTensor([idxs.size(0)])
        if split:
            return [self.trg.vocab.itos[i] for i in idxs[1:eos_idx]]
        else:
            return ' '.join([self.trg.vocab.itos[i] for i in idxs[1:eos_idx]])


"""
[Version Note]
    v0.1: basic implementation
"""

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    init_seed(1004)
    init_logging(LOGGER)
    LOGGER.info('COMMAND: {}'.format(' '.join(sys.argv)))

    # Raw data directory
    data_dir = os.path.join(os.path.expanduser('~'), 'datasets/wmt')
    wmt_dirs = {
        'train': os.path.join(data_dir, 'train/bitexts.selected'),
        'valid': os.path.join(data_dir, 'dev'),
        'test': os.path.join(data_dir, 'test')
    }

    # Directory to save preprocessed file
    save_dir = './data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_preprocess = False
    save_path = os.path.join(save_dir, 'wmt(small).pkl')

    # Save or load dataset
    dataset = WMTDataset()
    if save_preprocess:
        dataset.preprocess(wmt_dirs, save_path)
        LOGGER.info('## Saved datasets to %s' % save_path)
        save_path = None

    # Loader testing
    dataset.load(save_path)
    for batch in dataset.valid_iter:
        src, trg = batch.src[0], batch.trg[0]
        src_len, trg_len = batch.src[1], batch.trg[1]
        LOGGER.info('sample source: {}'.format(dataset.src_idx2word(src[0,:])))
        LOGGER.info('sample target: {}'.format(dataset.trg_idx2word(trg[0,:])))
        LOGGER.info('length of src: {}, trg: {}'.format(src_len[0], trg_len[0]))
        break

