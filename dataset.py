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

from utils import init_logging
from torchtext import data, datasets

LOGGER = logging.getLogger()


class WMTDataset(object):
    def __init__(self, wmt_dirs):
        self.initial_setting()
        self.process_wmt(wmt_dirs)
        print(vars(self))

    def initial_setting(self):
        LOGGER.info('Loading spacy (en, fr)')
        self.spacy_en = spacy.load('en')
        self.spacy_fr = spacy.load('fr')
        self.src = data.Field(tokenize=self.tokenize_en, init_token='<start>',
                              eos_token='<eos>')
        self.trg = data.Field(tokenize=self.tokenize_fr, init_token='<start>',
                              eos_token='<eos>')
        self.emb_dim = 50

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def tokenize_fr(self, text):
        return [tok.text for tok in self.spacy_fr.tokenizer(text)]

    def process_wmt(self, dirs):
        LOGGER.info('Preprocessing WMT data')
        paths = {
            'train': set(),
            'valid': set(),
            'test': set()
        }
        self.results = {
            'train': [],
            'valid': [],
            'test': []
        }

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
        f = lambda ex: len(ex.src) <= 10 and len(ex.trg) <= 10
        for train_path in paths['train']:
            LOGGER.info('Train data: {}'.format(train_path))
            break
            
            train = datasets.TranslationDataset(
                exts=('.en', '.fr'), fields=(self.src, self.trg),
                path=train_path,
                filter_pred=f
            )
            self.results['train'].append(train)

        for valid_path in paths['valid']:
            LOGGER.info('Valid data: {}'.format(valid_path))
            valid = datasets.TranslationDataset(
                exts=('.en', '.fr'), fields=(self.src, self.trg),
                path=valid_path,
                filter_pred=f
            )
            self.results['valid'].append(valid)

        for test_path in paths['test']:
            LOGGER.info('Test data: {}'.format(test_path))
            test = datasets.TranslationDataset(
                exts=('.en', '.fr'), fields=(self.src, self.trg),
                path=test_path,
                filter_pred=f
            )
            self.results['test'].append(test)

    def prepare_iterator(self): 
        LOGGER.info('Building vocabulary using train data')
        self.src.build_vocab(self.results['valid'][0].src)
        self.trg.build_vocab(self.results['valid'][0].trg)
        self.src_vocab_size = len(self.src.vocab.itos)
        self.trc_vocab_size = len(self.trg.vocab.itos)

        train_iter = data.BucketIterator(
            self.results['valid'][0], batch_size=16,
            shuffle=True, repeat=True, device=-1,
        )
        valid_iter = data.BucketIterator(
            self.results['valid'][0], batch_size=16,
            shuffle=False, repeat=True, device=-1,
        )
        test_iter = data.BucketIterator(
            self.results['test'][0], batch_size=16,
            shuffle=False, repeat=True, device=-1,
        )
        self.train_iter = iter(train_iter)
        self.valid_iter = iter(valid_iter)
        self.test_iter = iter(test_iter)
        
        '''
        LOGGER.info('Iterator test')
        sample = next(self.test_iter)
        print(self.src_idx2word(sample.src[:,0]))
        print(self.trg_idx2word(sample.trg[:,0]))
        exit()
        '''

    def get_src_vocab_vectors(self):
        return self.src.vocab.vectors

    def get_src_vocab_vectors(self):
        return self.trg.vocab.vectors

    def next_batch(self, mode, gpu=False):
        if mode == 'train':
            batch = next(self.train_iter)
        elif mode == 'valid':
            batch = next(self.valid_iter)
        elif mode == 'test':
            batch = next(self.test_iter)
        else:
            raise NotImplementedError

        if gpu:
            return batch.src.cuda(), batch.trg.cuda()

        return batch.src, batch.trg

    def src_idx2word(self, idxs, split=False):
        eos_idx = torch.argmax(idxs == self.src.vocab.stoi['<eos>'])
        if split:
            return [self.src.vocab.itos[i] for i in idxs[:eos_idx+1]]
        else:
            return ' '.join([self.src.vocab.itos[i] for i in idxs[:eos_idx+1]])

    def trg_idx2word(self, idxs, split=False):
        eos_idx = torch.argmax(idxs == self.trg.vocab.stoi['<eos>'])
        if eos_idx == 0:
            eos_idx = torch.LongTensor([-1])
        if split:
            return [self.trg.vocab.itos[i] for i in idxs[:eos_idx+1]]
        else:
            return ' '.join([self.trg.vocab.itos[i] for i in idxs[:eos_idx+1]])



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

    save_preprocess = True
    save_path = os.path.join(save_dir, 'wmt(tmp).pkl')
    load_path = os.path.join(save_dir, 'wmt(tmp).pkl')

    # Save or load dataset
    if save_preprocess:
        dataset = WMTDataset(wmt_dirs)
        pickle.dump(dataset, open(save_path, 'wb'))
        print('## Save preprocess %s' % save_path)
    else:
        print('## Load preprocess %s' % load_path)
        dataset = pickle.load(open(load_path, 'rb'))
   
    # Loader testing
    dataset.prepare_iterator()
    for src, trg in dataset.next_batch('valid'):
        print(dataset.src_idx2word(src[:,0]))
        print(dataset.trg_idx2word(trg[:,0]))
        break

