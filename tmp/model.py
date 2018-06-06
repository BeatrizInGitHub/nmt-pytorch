import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain


class SEQ2SEQ(nn.Module):
    """
    Bahdanau et al. "Neural Machine Translation by Jointly Learning to 
                     Align and Translate" ICLR. 2015.
    """

    def __init__(self, source_n_vocab, target_n_vocab, h_dim, emb_dim,
                 p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, 
                 eos_idx=3, freeze_embeddings=False, gpu=False):

        super(SEQ2SEQ, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx

        self.source_n_vocab = source_n_vocab
        self.target_n_vocab = target_n_vocab
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.p_word_dropout = p_word_dropout
        self.num_layers = 1
        self.num_experts = 1

        self.gpu = gpu

        """
        Word embeddings layer
        """
        self.src_emb = nn.Embedding(source_n_vocab, emb_dim, self.PAD_IDX)
        self.trg_emb = nn.Embedding(target_n_vocab, emb_dim, self.PAD_IDX)

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.emb_dim, h_dim,
                              num_layers=self.num_layers)

        """
        Decoder is GRU with `z` and `c` appended at its inputs
        """
        self.decoder = nn.GRU(self.emb_dim, h_dim,
                              num_layers=self.num_layers)
        self.decoder_fc = nn.Linear(h_dim, target_n_vocab) 
        # self.emb_fc = nn.ModuleList([nn.Linear(z_dim+c_dim, self.emb_dim)]*self.num_experts)
        # self.coef_fc = nn.Linear(z_dim+c_dim, self.num_experts)

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = chain(
            self.encoder.parameters(),
        )

        self.decoder_params = chain(
            self.decoder.parameters(), self.decoder_fc.parameters(), 
            # self.emb_fc.parameters()
        )

        self.s2s_params = chain(
            self.encoder_params, self.decoder_params, 
            self.src_emb.parameters(), self.trg_emb.parameters(),
        )
        self.s2s_params = filter(lambda p: p.requires_grad, self.s2s_params)

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """
        inputs = self.src_emb(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, h = self.encoder(inputs, None)

        return h

    def forward_decoder(self, inputs, hidden):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)
        mbsize = dec_inputs.size(1)

        # 1 x mbsize x (z_dim+c_dim)
        inputs_emb = self.trg_emb(dec_inputs)  # seq_len x mbsize x emb_dim

        outputs, hidden = self.decoder(inputs_emb, hidden)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len*mbsize, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, mbsize, self.target_n_vocab)

        # return outputs => word
        # mixture = self.emb_fc[0](outputs)
        # emb_out = [F.tanh(fc(outputs)) for fc in self.emb_fc]
        # coef_out = F.softmax(self.coef_fc(outputs), 1)
        # mixture = sum([single_expert * coef_out[:,k].unsqueeze(1) 
        #                for k, single_expert in enumerate(emb_out)])

        return y, hidden

    def forward(self, sentence, use_c_prior=True):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """
        self.train()

        seq_len, mbsize = sentence.size()

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        hidden = self.forward_encoder(enc_inputs)

        # Decoder: sentence -> y
        y, emb_out = self.forward_decoder(dec_inputs, hidden)
        # emb_target = Variable(self.word_emb(dec_targets)).cuda()

        # emb_loss = torch.sum(F.cosine_similarity(
        #     emb_out.view(-1, self.emb_dim),
        #     emb_target.view(-1, self.emb_dim)
        # ))
        loss = F.cross_entropy(
            y.view(-1, self.target_n_vocab), dec_targets.view(-1), size_average=True
        )
        return loss

    def translate_sentences(self, inputs):
        """
        Translate sentences of (mbsize x max_sent_len)
        """
        samples = []
        mbsize = inputs.size(1)
        hidden = self.forward_encoder(inputs)

        for k in range(mbsize):
            samples.append(self.translate_sentence(hidden[:,k,:], 
                                                   inputs.size(0)))

        return torch.LongTensor(samples).cuda()

    def translate_sentence(self, hidden, max_sent_len):
        """
        Translate a single sentence.
        """
        self.eval()

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        hidden = hidden.unsqueeze(1)
        outputs = []

        for i in range(max_sent_len):
            emb = self.trg_emb(word).view(1, 1, -1)

            output, hidden = self.decoder(emb, hidden)
            y = self.decoder_fc(output).view(-1)

            # New embed
            # output = output.squeeze(0)
            # emb = self.emb_fc[0](output)
            # emb_out = [F.tanh(fc(output)) for fc in self.emb_fc]
            # coef_out = F.softmax(self.coef_fc(output), 1)
            
            # emb = sum([single_expert * coef_out[:,k].unsqueeze(1) 
            #            for k, single_expert in enumerate(emb_out)])

            # idx = torch.sum(F.mse_loss(emb, self.word_emb.weight, reduce=False), dim=1)
            # idx = F.cosine_similarity(emb.view(-1, self.emb_dim), self.word_emb.weight)
            # idx = torch.argmin(idx)
            # emb = emb.unsqueeze(0)

            y = F.softmax(y, dim=0)
            idx = torch.multinomial(y, 1)
            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            # if idx == self.EOS_IDX:
            #     break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        return outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)
