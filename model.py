import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import math
import sys
import os
import logging

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.parameter import Parameter

LOGGER = logging.getLogger(__name__)


class Seq2SeqAttModel(nn.Module):
    def __init__(self, rnn_dim, rnn_layer, rnn_dropout, bi_rnn, pad_idx,
            word_embed_dim, align_dim, maxout_dim, src_vocab_size, 
            trg_vocab_size, linear_dropout, learning_rate):

        super(Seq2SeqAttModel, self).__init__()

        # Save model configs
        self.rnn_dim = rnn_dim
        self.rnn_layer = rnn_layer
        self.rnn_dropout = rnn_dropout
        self.bi_rnn = bi_rnn
        self.word_embed_dim = word_embed_dim
        self.align_dim = align_dim
        self.maxout_dim = maxout_dim
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.linear_dropout = linear_dropout

        # Parameters for word embeddings
        self.src_word_embed = nn.Embedding(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=pad_idx
        )
        self.trg_word_embed = nn.Embedding(
            num_embeddings=self.trg_vocab_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=pad_idx
        )

        # Parameters for RNNs
        self.encoder = nn.GRU(
            input_size=self.word_embed_dim, 
            hidden_size=self.rnn_dim, 
            num_layers=self.rnn_layer,
            dropout=self.rnn_dropout,
            bidirectional=bi_rnn,
            batch_first=True
        )
        self.decoder = nn.GRUCell(
            input_size=self.word_embed_dim, 
            hidden_size=self.rnn_dim, 
        )
        
        # Parameters for alignment
        self.w_s = nn.Linear(self.rnn_dim, self.rnn_dim)
        self.v_a = nn.Linear(self.align_dim, 1)
        self.w_a = nn.Linear(self.rnn_dim, self.align_dim)
        self.u_a = nn.Linear(self.rnn_dim * 2, self.align_dim)

        # Parameters for maxout
        self.w_o = nn.Linear(self.maxout_dim, self.trg_vocab_size)
        self.u_o = nn.Linear(self.rnn_dim, self.maxout_dim * 2)
        self.v_o = nn.Linear(self.word_embed_dim, self.maxout_dim * 2)
        self.c_o = nn.Linear(self.rnn_dim * 2, self.maxout_dim * 2)

        # Print model info and set optimizer, loss
        info, self.params = self.get_model_params()
        LOGGER.info(info)
        self.optimizer = optim.Adadelta(self.params, learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def init_h(self, batch_size):
        return Variable(torch.zeros(self.rnn_layer * (2 if self.bi_rnn else 1),
                        batch_size, self.rnn_dim)).cuda()

    def encode(self, inputs, length):
        w_embed = self.src_word_embed(inputs)
        batch_size = inputs.size(0)
        maxlen = inputs.size(1)

        if False:
            # Sort inputs
            _, sort_idx = torch.sort(length, dim=0, descending=True)
            _, unsort_idx = torch.sort(sort_idx, dim=0)

            # Pack padded sequence
            w_embed = w_embed.index_select(0, Variable(sort_idx).cuda())
            sorted_len = length.index_select(0, sort_idx).tolist()
            w_packed = pack_padded_sequence(w_embed, sorted_len, batch_first=True) 
        else:
            w_packed = w_embed

        # Run LSTM
        init_h = self.init_h(batch_size)
        outputs, last_state = self.encoder(w_packed, init_h)

        if False:
            # TODO: pad packed sequence
            # Unsort hidden states
            outputs = outputs.index_select(0, Variable(unsort_idx).cuda())
        else:
            outputs = outputs
        
        return outputs, last_state
    
    def decode(self, inputs, length, annotations, encoder_state):
        w_embed = self.trg_word_embed(inputs)
        batch_size = inputs.size(0)
        maxlen = inputs.size(1)

        outputs = []
        hidden = encoder_state[1,:,:]
        for time_step in range(maxlen):

            # Compute alignments, context
            a_list = []
            for k in range(annotations.size(1)):
                a_ik = self.v_a(F.tanh(self.w_a(hidden) + 
                    self.u_a(annotations[:,k,:])))
                a_list.append(a_ik)
            alignment = torch.cat(a_list, dim=1)
            alignment = F.softmax(alignment, dim=1).unsqueeze(2)
            context = torch.sum(alignment * annotations, dim=1)

            # Maxout layer
            t_i = self.u_o(hidden) + self.v_o(w_embed[:,time_step,:]) \
                  + self.c_o(context)
            t_i = torch.max(t_i.view(-1, self.maxout_dim, 2), dim=-1)[0]
            out = self.w_o(t_i)

            # Update hidden
            hidden = self.decoder(w_embed[:,time_step,:], hidden)
            outputs.append(out) 

        outputs = torch.stack(outputs).transpose(0, 1)
        return outputs
    
    def forward(self, src, src_len, trg, trg_len):
        annotations, last_state = self.encode(src, src_len)
        outputs = self.decode(trg[:,:-1], trg_len, annotations, last_state)
        return outputs
    
    def get_loss(self, outputs, targets):
        outputs = outputs.contiguous().view(-1, self.trg_vocab_size)
        targets = targets.contiguous().view(-1)
        return self.criterion(outputs, targets)

    def get_model_params(self):
        params = []
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())

        return '{}, param size: {:,}'.format(self, total_size), params

    def save_checkpoint(self, state, checkpoint_dir, filename):
        filename = os.path.join(checkpoint_dir, filename)
        LOGGER.info('Save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_dir, filename):
        filename = os.path.join(checkpoint_dir, filename)
        LOGGER.info('Load checkpoint %s' % filename)
        checkpoint = torch.load(filename)

        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

