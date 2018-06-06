from torchtext import data, datasets
from torchtext.vocab import GloVe

import torch
import os
import spacy
import re


data_path = os.path.join(os.path.expanduser("~"), 'datasets')

class WMT_Dataset:

    def __init__(self, emb_dim=300, mbsize=32):
        self.spacy_en = spacy.load('en')
        self.spacy_fr = spacy.load('fr')
        self.url = re.compile('(<url>.*</url>)')

        self.SOURCE = data.Field(tokenize=self.tokenize_en, init_token='<start>',
                                 eos_token='<eos>')
        self.TARGET = data.Field(tokenize=self.tokenize_fr, init_token='<start>',
                                 eos_token='<eos>')

        f = lambda ex: len(ex.src) <= 30 and len(ex.trg) <= 30

        train1, val, test = datasets.TranslationDataset.splits(
            exts=('.en', '.fr'), fields=(self.SOURCE, self.TARGET),
            path=os.path.join(data_path, 'wmt'),
            # train='train/bitexts.selected/dev08_11',
            train='train/bitexts.selected/un2000_pc34',
            validation='dev/ntst1213',
            test='test/ntst14',
            filter_pred=f
        )
        
        self.SOURCE.build_vocab(train1.src, min_freq=3)
        self.TARGET.build_vocab(train1.trg, min_freq=3)
        self.source_n_vocab = len(self.SOURCE.vocab.itos)
        self.target_n_vocab = len(self.TARGET.vocab.itos)
        self.emb_dim = emb_dim

        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train1, val, test), batch_size=mbsize, 
            shuffle=True, repeat=True, device=-1,
        )
        
        self.train_iter = iter(train_iter)
        self.val_iter = iter(val_iter)
        self.test_iter = iter(test_iter)

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(self.url.sub('@URL@', text))]

    def tokenize_fr(self, text):
        return [tok.text for tok in self.spacy_fr.tokenizer(self.url.sub('@URL@', text))]

    def get_source_vocab_vectors(self):
        return self.SOURCE.vocab.vectors

    def get_target_vocab_vectors(self):
        return self.TARGET.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.src.cuda(), batch.trg.cuda()

        return batch.src, batch.trg

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.src.cuda(), batch.trg.cuda()

        return batch.src, batch.trg

    def next_test_batch(self, gpu=False):
        batch = next(self.test_iter)

        if gpu:
            return batch.src.cuda(), batch.trg.cuda()

        return batch.src, batch.trg

    def source_idxs2sentence(self, idxs, split=False):
        eos_idx = torch.argmax(idxs == self.SOURCE.vocab.stoi['<eos>'])
        if split:
            return [self.SOURCE.vocab.itos[i] for i in idxs[:eos_idx+1]]
        else:
            return ' '.join([self.SOURCE.vocab.itos[i] for i in idxs[:eos_idx+1]])

    def target_idxs2sentence(self, idxs, split=False):
        eos_idx = torch.argmax(idxs == self.TARGET.vocab.stoi['<eos>'])
        if eos_idx == 0:
            eos_idx = torch.LongTensor([-1])
        if split:
            return [self.TARGET.vocab.itos[i] for i in idxs[:eos_idx+1]]
        else:
            return ' '.join([self.TARGET.vocab.itos[i] for i in idxs[:eos_idx+1]])

