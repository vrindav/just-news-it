#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys
import re
import glob

#reload(sys)
#sys.setdefaultencoding('utf8')

import sys
sys.path.insert(0, '../')

import os
import time
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from data_util.batcher import Batcher, Example, Batch
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from train_util import get_input_from_batch

from model import Model

sys.path.insert(0, './transformer_model/')
from Beam import Beam


use_cuda = config.use_gpu and torch.cuda.is_available()

def write_results(decoded_words, ex_index, _rouge_dec_dir):
  decoded_sents = []
  while len(decoded_words) > 0:
    try:
      fst_period_idx = decoded_words.index(".")
    except ValueError:
      fst_period_idx = len(decoded_words)
    sent = decoded_words[:fst_period_idx + 1]
    decoded_words = decoded_words[fst_period_idx + 1:]
    decoded_sents.append(' '.join(sent))

  # pyrouge calls a perl script that puts the data into HTML files.
  # Therefore we need to make our output HTML safe.
  print(decoded_sents)
  decoded_sents = [make_html_safe(w) for w in decoded_sents]

  print(decoded_sents)

  decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

  with open(decoded_file, "w") as f:
    for idx, sent in enumerate(decoded_sents):
      f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

class Summarizer(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        '''
        opt needs to contain:
            - model_file_path
            - n_best
            - max_token_seq_len
        '''
        self.opt = opt
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        print("Max article len", config.max_article_len)
        model = Model(config.vocab_size, config.vocab_size, config.max_article_len)

        checkpoint = torch.load(opt["model_file_path"], map_location= lambda storage, location: storage)

        # model saved as: 
        # state = {
        #     'iter': iter,
        #     'transformer_state_dict': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'current_loss': running_avg_loss
        # }

            
        model.load_state_dict(checkpoint['transformer_state_dict'])

        print('[Info] Trained model state loaded.')

        #model.word_prob_prj = nn.LogSoftmax(dim=1)

        self.model = model.to(self.device)

        self.model.eval()

        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (opt["model_file_path"].split("/")[-1]))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batches = self.read_opeds(config.oped_data_path, self.vocab, config.beam_size)

        #time.sleep(15)
        
        print('[Info] Summarizer object created.')

    def read_opeds(self, config_path, vocab, beam_size):
        file_list = glob.glob(config_path)
        #file_list = os.listdir(config_path)
        batch_list = []
        for file in file_list:
            with open(file, 'rb') as f:
                text = f.read().lower().decode('utf-8')
                text = re.sub('\n', '', text)
                text = re.sub(r'([.,!?()"])', r' \1 ', text).encode('utf-8')
                #print(text)

                ex = Example(text, [], vocab)

                # text = text.split()
                # if len(text) > config.max_enc_steps:
                #     text = text[:config.max_enc_steps]
                # enc_input = [vocab.word2id(w.decode('utf-8')) for w in text]
                # assert(sum(enc_input) != 0)

                enc_input = [ex for _ in range(beam_size)]
                batch = Batch(enc_input, vocab, beam_size)
                batch_list.append(batch)
                print(batch.enc_batch)
        return batch_list

    def summarize_batch(self, src_seq, src_pos):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
                dec_output, *_ = self.model.transformer.decoder(dec_seq, dec_pos, src_seq, enc_output)
                
                # print("dec_output (line 136)", dec_output.size())
                # print("batch size", config.batch_size)
                # print("decoder output", dec_output[:, -1, :10])

                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                logits = self.model.transformer.tgt_word_prj(dec_output)

                # print("logits size", logits.size())
                # print("logits", logits[:, :10])

                word_prob = logits#F.softmax(logits, dim=1)

                # print(word_prob[:, :10])
                # print("word_prob", torch.max(word_prob, 1))

                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)

            #print("first dec_seq", dec_seq)
            # print("first dec_pos", dec_pos)

            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Encode
            # print("src_seq", src_seq.size())
            # print("src_pos", src_pos.size())

            #print("src_seq", src_seq[:, :20])
            #print("src_pos", src_pos[:, :20])

            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            src_enc, *_ = self.model.transformer.encoder(src_seq, src_pos)

            #-- Repeat data for beam search
            #n_bm = config.beam_size
            n_bm = 1
            n_inst, len_s, d_h = src_enc.size()

            # print("src_enc_shape", src_enc.size())

            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.opt["max_token_seq_len"] + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.opt["n_best"])

        return batch_hyp, batch_scores

    def get_pos_data(self, padding_masks):
        batch_size, seq_len = padding_masks.shape

        pos_data = [[ j + 1 if padding_masks[i][j] == 1 else 0 for j in range(seq_len)] for i in range(batch_size)]

        pos_data = torch.tensor(pos_data, dtype=torch.long)

        if use_cuda:
            pos_data = pos_data.cuda()

        return pos_data

    def decode(self):
        start = time.time()
        counter = 0
        #batch = self.batcher.next_batch()
        #print(batch.enc_batch)

        keep = True
        for batch in self.batches:
            #keep = False # one batch only

            # Run beam search to get best Hypothesis
            enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = get_input_from_batch(batch, use_cuda)

            enc_batch = enc_batch[0:1, :]
            enc_padding_mask = enc_padding_mask[0:1, :]

            in_seq = enc_batch
            in_pos = self.get_pos_data(enc_padding_mask)
            #print("enc_padding_mask", enc_padding_mask)

            #print("Summarizing one batch...")

            batch_hyp, batch_scores = self.summarize_batch(in_seq, in_pos)

            # Extract the output ids from the hypothesis and convert back to words
            output_words = np.array(batch_hyp)
            output_words = output_words[:, 0, 1:]

            for i, out_sent in enumerate(output_words):

                decoded_words = data.outputids2words(out_sent, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

                original_abstract_sents = batch.original_abstracts_sents[i]

                write_for_rouge(original_abstract_sents, decoded_words, counter,
                                self._rouge_ref_dir, self._rouge_dec_dir)
                counter += 1

            if counter % 1 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            #batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        # print("Now starting ROUGE eval...")
        # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        # rouge_log(results_dict, self._decode_dir)



if __name__ == '__main__':
    model_filename = sys.argv[1]
    opt = {}

    opt["model_file_path"] = model_filename
    opt["n_best"] = 3
    opt["max_token_seq_len"] = 200

    summarizer = Summarizer(opt)
    summarizer.decode()


