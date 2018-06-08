import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
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
            input_size=self.word_embed_dim + self.rnn_dim * 2, 
            hidden_size=self.rnn_dim, 
        )
        
        # Parameters for alignment
        self.w_s = nn.Linear(self.rnn_dim, self.rnn_dim)
        self.v_a = nn.Linear(self.align_dim, 1, bias=False)
        self.w_a = nn.Linear(self.rnn_dim, self.align_dim)
        self.u_a = nn.Linear(self.rnn_dim * 2, self.align_dim)

        # Parameters for maxout
        self.w_o = nn.Linear(self.maxout_dim, self.trg_vocab_size)
        self.u_o = nn.Linear(self.rnn_dim, self.maxout_dim * 2)
        self.v_o = nn.Linear(self.word_embed_dim, self.maxout_dim * 2)
        self.c_o = nn.Linear(self.rnn_dim * 2, self.maxout_dim * 2)

        # Print model info and set optimizer, loss, initialization
        info, self.params = self.get_model_params()
        LOGGER.info(info)
        self.optimizer = optim.Adadelta(self.params, learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.init_weights()

    def init_weights(self):
        # Orthogonal/zero initialization for RNNs
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            if 'bias' in name:
                param.data.zero_()
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            if 'bias' in name:
                param.data.zero_()

        # N(0, 0.001^2) for W_a, U_a
        nn.init.normal_(self.w_a.weight, mean=0, std=0.001)
        nn.init.normal_(self.u_a.weight, mean=0, std=0.001)
        self.w_a.bias.data.zero_()
        self.u_a.bias.data.zero_()

        # V_a to zero
        for param in self.v_a.parameters():
            param.data.zero_()

        # Others to N(0, 0.01^2)
        nn.init.normal_(self.w_s.weight, mean=0, std=0.01)
        nn.init.normal_(self.w_o.weight, mean=0, std=0.01)
        nn.init.normal_(self.u_o.weight, mean=0, std=0.01)
        nn.init.normal_(self.v_o.weight, mean=0, std=0.01)
        nn.init.normal_(self.c_o.weight, mean=0, std=0.01)
        self.w_s.bias.data.zero_()
        self.w_o.bias.data.zero_()
        self.u_o.bias.data.zero_()
        self.v_o.bias.data.zero_()
        self.c_o.bias.data.zero_()

    def init_h(self, batch_size):
        return Variable(torch.zeros(self.rnn_layer * (2 if self.bi_rnn else 1),
                        batch_size, self.rnn_dim)).cuda()

    def encode(self, inputs, length):
        w_embed = self.src_word_embed(inputs)
        batch_size = inputs.size(0)
        maxlen = inputs.size(1)

        # Run GRU
        init_h = self.init_h(batch_size)
        outputs, last_state = self.encoder(w_embed, init_h)
        
        return outputs, last_state

    def decode_step(self, inp_embed, hidden, annotations):
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
        t_i = self.u_o(hidden) + self.v_o(inp_embed) + self.c_o(context)
        t_i = torch.max(t_i.view(-1, self.maxout_dim, 2), dim=-1)[0]
        out = self.w_o(t_i)

        # Update hidden
        hidden = self.decoder(torch.cat((inp_embed, context), dim=-1), hidden)
        return out, hidden
    
    def decode(self, inputs, length, annotations, encoder_state):
        w_embed = self.trg_word_embed(inputs)
        batch_size = inputs.size(0)
        maxlen = inputs.size(1)

        outputs = []
        hidden = encoder_state[1,:,:]
        for time_step in range(maxlen):
            out, hidden = self.decode_step(w_embed[:,time_step,:], 
                hidden, annotations)
            outputs.append(out) 

        outputs = torch.stack(outputs).transpose(0, 1)
        return outputs
    
    def beam_search(self, start, maxlen, annotations, encoder_state):
        batch_size = start.size(0)
        beam_size = 1
        # best_seq = [[[start[0].tolist()]]]*batch_size

        outputs = []
        hidden = encoder_state[1,:,:]

        # Beam search TODO: beam size > 1
        inp = start
        for time_step in range(maxlen):
            # inp = torch.cuda.LongTensor([s[-1] for q in best_seq for s in q]).view(-1)
            w_embed = self.trg_word_embed(inp)
            out, hidden = self.decode_step(w_embed, hidden, annotations)
            cand_score, inp = torch.topk(out, beam_size, dim=-1)
            inp.squeeze_(1)

            # for b_idx, b_seqs in enumerate(best_seq):
            #     for c_idx, cand in enumerate(cand_inp[b_idx]):
            outputs.append(out) 

        outputs = torch.stack(outputs).transpose(0, 1)
        return outputs
    
    def forward(self, src, src_len, trg, trg_len):
        annotations, last_state = self.encode(src, src_len)
        if self.training:
            outputs = self.decode(trg[:,:-1], trg_len, annotations, last_state)
        else:
            outputs = self.beam_search(trg[:,0], trg.size(1)-1, annotations,
                last_state)

        return outputs
    
    def get_loss(self, outputs, targets, length):
        batch_size = outputs.size(0)
        maxlen = outputs.size(1)

        # Create mask
        mask = torch.arange(maxlen).unsqueeze(0).repeat(batch_size, 1).long().cuda()
        mask = mask < length.unsqueeze(1).repeat(1, maxlen)

        # Gather by mask
        outputs = outputs.contiguous().view(-1, self.trg_vocab_size)
        targets = targets.contiguous().view(-1)
        losses = self.criterion(outputs, targets)
        losses = losses.view(batch_size, -1) * mask.float() 

        return torch.sum(losses) / torch.sum(mask.float())

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

