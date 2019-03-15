from __future__ import unicode_literals, print_function, division

import sys
sys.path.insert(0, '../')
import os
import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adam, Adagrad

from data_util import config

from transformer_model.Optim import ScheduledOptim
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'transformer_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer._optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

        return model_save_path

    def setup_train(self, n_src_vocab, n_tgt_vocab, model_file_path=None):
        self.model = Model(n_src_vocab, n_tgt_vocab, config.max_article_len)

        params = list(self.model.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = ScheduledOptim(Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
            config.d_model, config.n_warmup_steps)
        #self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index = 1)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            self.model.load_state_dict(state['transformer_state_dict'])

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def get_pos_data(self, padding_masks):
        batch_size, seq_len = padding_masks.shape

        pos_data = [[ j + 1 if padding_masks[i][j] == 1 else 0 for j in range(seq_len)] for i in range(batch_size)]

        pos_data = torch.tensor(pos_data, dtype=torch.long)

        if use_cuda:
            pos_data = pos_data.cuda()

        return pos_data

    def train_one_batch(self, batch, iter):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)
       # print(target_batch[:, 1:].contiguous().view(-1)[-10:])
        #print(dec_batch[:, 1:].contiguous().view(-1)[-10:])

        in_seq = enc_batch
        in_pos = self.get_pos_data(enc_padding_mask)
        tgt_seq = dec_batch
        tgt_pos = self.get_pos_data(dec_padding_mask)
        
        # padding is already done in previous function (see batcher.py - init_decoder_seq & init_decoder_seq - Batch class)
        self.optimizer.zero_grad()
        #logits = self.model.forward(in_seq, in_pos, tgt_seq, tgt_pos)
        logits = self.model.forward(in_seq, in_pos, tgt_seq, tgt_pos, extra_zeros, enc_batch_extend_vocab)

        # compute loss from logits
        #loss = self.loss_func(logits, target_batch.contiguous().view(-1))

        losses = []
        for i in range(confi.batch_size):
            target = target_batch[i]
            ex_logits = logits[i]

            target[ex_logits == 0] = 0
            losses.append(self.loss_func(ex_logits, target))

        sum_losses = torch.mean(torch.stack(losses, 1), 1)

        if iter % 50 == 0 and False:
            print(iter, loss)
            print('\n')
            # print(logits.max(1)[1][:20])
            # print('\n')
            # print(target_batch.contiguous().view(-1)[:20])
            # print('\n')
            #print(target_batch.contiguous().view(-1)[-10:])

        loss.backward()

        #print(logits.max(1)[1])
        #print('\n')
        #print(tgt_seq[:, 1:].contiguous().view(-1)[:10])
        #print(tgt_seq[:, 1:].contiguous().view(-1)[-10:])
        
        self.norm = clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.parameters(), config.max_grad_norm)

        #self.optimizer.step()
        self.optimizer.step_and_update_lr()

        return loss.item()

    def trainIters(self, n_src_vocab, n_tgt_vocab, n_iters, model_file_path=None):

        print("Setting up the model...")

        iter, running_avg_loss = self.setup_train(n_src_vocab, n_tgt_vocab, model_file_path)

        print("Starting training...")
        
        start = time.time()

        #only_batch = None

        while iter < n_iters:
            batch = self.batcher.next_batch()
            
            # if iter == 0:
            #     only_batch = batch
            # else:
            #     batch = only_batch

            loss = self.train_one_batch(batch, iter)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1
            
            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 50
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            
            if iter % 5000 == 0:
                path = self.save_model(running_avg_loss, iter)

                print("Saving Checkpoint at {}".format(path))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    # Same vocabulary for both input and output
    n_src_vocab = config.vocab_size
    n_tgt_vocab = config.vocab_size

    train_processor = Train()

    ##### TODO: get the right arguments to pass into trainIters
    train_processor.trainIters(n_src_vocab, n_tgt_vocab, config.max_iterations, args.model_file_path)




