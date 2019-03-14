from __future__ import unicode_literals, print_function, division

import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import random

from transformer_model import Models as tm

from data_util import config

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# class Transformer(nn.Module): defined in transformer_model/Models.py

class Model(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, len_max_seq, model_file_path=None, is_eval=False):
        super(Model, self).__init__()

        transformer = tm.Transformer(n_src_vocab, n_tgt_vocab, len_max_seq, d_word_vec = config.d_word_vec, 
                        d_model = config.d_model, d_inner = config.d_inner, n_layers = config.n_layers, 
                        n_head = config.n_head, d_k = config.d_k, d_v = config.d_v, dropout = config.dropout)
        #transformer = tm.Transpointer(n_src_vocab, n_tgt_vocab, len_max_seq, n_head=8)

        if use_cuda:
            transformer = transformer.cuda()
            self.device = torch.device("cuda")

        self.transformer = transformer

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
    #def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, extra_zeros=None, enc_batch_extend_vocab=None):

        return self.transformer(src_seq, src_pos, tgt_seq, tgt_pos)
        #return self.transformer(src_seq, src_pos, tgt_seq, tgt_pos, extra_zeros, enc_batch_extend_vocab)


