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

        transformer = tm.Transformer(n_src_vocab, n_tgt_vocab, len_max_seq)

            # d_word_vec=512, d_model=512, d_inner=2048,
            # n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            # tgt_emb_prj_weight_sharing=True,
            # emb_src_tgt_weight_sharing=True

        if use_cuda:
            transformer = transformer.cuda()
            self.device = torch.device("cuda")

        self.transformer = transformer
        
        ####### TODO: load a pretrained model
        # if model_file_path is not None:
        #     state = torch.load(model_file_path, map_location= lambda storage, location: storage)
        #     self.encoder.load_state_dict(state['encoder_state_dict'])
        #     self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        return self.transformer(src_seq, src_pos, tgt_seq, tgt_pos)


