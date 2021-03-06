''' Define the Transformer model '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import transformer_model.Constants as Constants
from transformer_model.Layers import EncoderLayer, DecoderLayer

import sys
sys.path.insert(0, '../../')
from data_util import config

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
	assert seq.dim() == 2
	return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
	''' Sinusoid position encoding table '''

	def cal_angle(position, hid_idx):
		return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

	def get_posi_angle_vec(position):
		return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

	sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

	if padding_idx is not None:
		# zero vector for padding dimension
		sinusoid_table[padding_idx] = 0.

	return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
	''' For masking out the padding part of key sequence. '''

	# Expand to fit the shape of key query attention matrix.
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(Constants.PAD)
	padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

	return padding_mask

def get_subsequent_mask(seq):
	''' For masking out the subsequent info. '''

	sz_b, len_s = seq.size()
	subsequent_mask = torch.triu(
		torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
	subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

	return subsequent_mask

def get_local_mask(seq, window_size):
	''' For masking out the info. Not in the window.'''

	sz_b, len_s = seq.size()
	subsequent_mask = torch.triu(
		torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=window_size)
	subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

	prev_mask = torch.tril(
		torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=-window_size)
	prev_mask = prev_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

	return subsequent_mask + prev_mask

class Encoder(nn.Module):
	''' A encoder model with self attention mechanism. '''

	def __init__(
			self,
			n_src_vocab, len_max_seq, d_word_vec,
			n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1):

		super().__init__()

		n_position = len_max_seq + 1

		self.src_word_emb = nn.Embedding(
			n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

		self.position_enc = nn.Embedding.from_pretrained(
			get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
			freeze=True)

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])

	def forward(self, src_seq, src_pos, return_attns=False):

		enc_slf_attn_list = []

		# -- Prepare masks
		print_ = True
		slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
		if config.local_attention_window_size > 0:
			slf_attn_mask = (slf_attn_mask.int() | get_local_mask(src_seq, config.local_attention_window_size).int()).byte()

		non_pad_mask = get_non_pad_mask(src_seq)

		# -- Forward
		enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(
				enc_output,
				non_pad_mask=non_pad_mask,
				slf_attn_mask=slf_attn_mask)
			if return_attns:
				enc_slf_attn_list += [enc_slf_attn]

		if return_attns:
			return enc_output, enc_slf_attn_list

		return enc_output,

class Decoder(nn.Module):
	''' A decoder model with self attention mechanism. '''

	def __init__(
			self,
			n_tgt_vocab, len_max_seq, d_word_vec,
			n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1):

		super().__init__()
		n_position = len_max_seq + 1

		self.tgt_word_emb = nn.Embedding(
			n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

		self.position_enc = nn.Embedding.from_pretrained(
			get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
			freeze=True)

		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])

	def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False, return_dec_input=False):

		dec_slf_attn_list, dec_enc_attn_list = [], []

		# -- Prepare masks
		non_pad_mask = get_non_pad_mask(tgt_seq)

		slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
		slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
		slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

		dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

		# -- Forward
		dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
		dec_input = dec_output.clone()

		for dec_layer in self.layer_stack:
			dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
				dec_output, enc_output,
				non_pad_mask=non_pad_mask,
				slf_attn_mask=slf_attn_mask,
				dec_enc_attn_mask=dec_enc_attn_mask)

			if return_dec_input:
				dec_slf_attn_list += [dec_slf_attn]
				dec_enc_attn_list += [dec_enc_attn]

		#if return_attns:
		#    return dec_output, dec_slf_attn_list, dec_enc_attn_list
		if return_dec_input:
			return dec_output, dec_input, dec_enc_attn_list[-1]
		return dec_output, 

class Transformer(nn.Module):
	''' A sequence to sequence model with attention mechanism. '''

	def __init__(
			self,
			n_src_vocab, n_tgt_vocab, len_max_seq,
			d_word_vec=512, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
			tgt_emb_prj_weight_sharing=True,
			emb_src_tgt_weight_sharing=True):

		super().__init__()

		self.encoder = Encoder(
			n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			dropout=dropout)

		self.decoder = Decoder(
			n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			dropout=dropout)

		self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
		nn.init.xavier_normal_(self.tgt_word_prj.weight)

		assert d_model == d_word_vec, \
		'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'

		if tgt_emb_prj_weight_sharing:
			# Share the weight matrix between target word embedding & the final logit dense layer
			self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
			self.x_logit_scale = (d_model ** -0.5)
		else:
			self.x_logit_scale = 1.

		if emb_src_tgt_weight_sharing:
			# Share the weight matrix between source & target word embeddings
			assert n_src_vocab == n_tgt_vocab, \
			"To share word embedding table, the vocabulary size of src/tgt shall be the same."
			self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

	def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

		batch_size, max_seq_len = src_seq.size()

		#tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

		enc_output, *_ = self.encoder(src_seq, src_pos)
		dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
				
		seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

		return seq_logit.view(-1, seq_logit.size(2))

class Transpointer(nn.Module):
	def __init__(
			self,
			n_src_vocab, n_tgt_vocab, len_max_seq,
			d_word_vec=512, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
			tgt_emb_prj_weight_sharing=True,
			emb_src_tgt_weight_sharing=True):

		super().__init__()

		self.n_head = n_head

		self.encoder = Encoder(
			n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			dropout=dropout)

		self.decoder = Decoder(
			n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			dropout=dropout)

		self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
		nn.init.xavier_normal_(self.tgt_word_prj.weight)

		assert d_model == d_word_vec, \
		'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'

		if tgt_emb_prj_weight_sharing:
			# Share the weight matrix between target word embedding & the final logit dense layer
			self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
			self.x_logit_scale = (d_model ** -0.5)
		else:
			self.x_logit_scale = 1.

		if emb_src_tgt_weight_sharing:
			# Share the weight matrix between source & target word embeddings
			assert n_src_vocab == n_tgt_vocab, \
			"To share word embedding table, the vocabulary size of src/tgt shall be the same."
			self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

		self.p_gen_linear = nn.Linear(config.max_dec_steps * 2 + config.max_article_len, 1)

	def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, extra_zeros, enc_batch_extend_vocab):

		batch_size, max_seq_len = src_seq.size()

		#tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

		enc_output, *_ = self.encoder(src_seq, src_pos)
		dec_output, dec_input, attn_dist = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, return_dec_input = True)
		
		# Reshape attn_dist to be batch_size by max_article_len
		attn_dist = attn_dist.reshape(self.n_head, config.batch_size, config.max_dec_steps, config.max_article_len) # -1 because of decoder shift
		attn_dist = attn_dist.permute(1, 0, 2, 3)

		# TODO: make this a linear layer
		attn_dist = torch.sum(attn_dist, dim=1)
		attn_dist = attn_dist.reshape(-1, attn_dist.size(2))

		concat = torch.cat((enc_output, dec_output, dec_input), dim = 1)
		concat = torch.mean(concat, dim=2)
		p_gen = self.p_gen_linear(concat)
		p_gen = torch.sigmoid(p_gen)
		p_gen = p_gen.repeat(1, config.max_dec_steps).reshape(-1, 1)

		seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
		seq_logit = seq_logit.view(-1, seq_logit.size(2))
		vocab_dist_ = p_gen * seq_logit
		attn_dist_ = (1 - p_gen) * attn_dist

		if extra_zeros is not None:
			extra_zeros = extra_zeros.repeat(1, config.max_dec_steps).reshape(-1, extra_zeros.size(1))
			vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

		enc_batch_extend_vocab = enc_batch_extend_vocab.repeat(1, config.max_dec_steps).reshape(-1, config.max_article_len)
		final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

		return final_dist

class ExtractiveTransformer(nn.Module):
	''' A sequence to sequence model with attention mechanism. '''

	def __init__(
			self,
			n_src_vocab, n_tgt_vocab, len_max_seq,
			d_word_vec=512, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
			tgt_emb_prj_weight_sharing=True,
			emb_src_tgt_weight_sharing=True):

		super().__init__()

		self.encoder = Encoder(
			n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			dropout=dropout)

		self.decoder = Decoder(
			n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			dropout=dropout)

		self.tgt_word_prj = nn.Linear(d_model, len_max_seq, bias=False)
		nn.init.xavier_normal_(self.tgt_word_prj.weight)

		self.n_tgt_vocab = n_tgt_vocab

		assert d_model == d_word_vec, \
		'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'

		if tgt_emb_prj_weight_sharing:
			# Share the weight matrix between target word embedding & the final logit dense layer
			self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
			self.x_logit_scale = (d_model ** -0.5)
		else:
			self.x_logit_scale = 1.

		if emb_src_tgt_weight_sharing:
			# Share the weight matrix between source & target word embeddings
			assert n_src_vocab == n_tgt_vocab, \
			"To share word embedding table, the vocabulary size of src/tgt shall be the same."
			self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

	def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, extra_zeros, enc_batch_extend_vocab):

		batch_size, max_seq_len = src_seq.size()

		#tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

		enc_output, *_ = self.encoder(src_seq, src_pos)
		dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
				
		seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

		#vocab_dist_ = torch.zeros(config.batch_size, config.max_dec_steps, self.n_tgt_vocab).cuda()
		initial_dist_ = Variable(torch.zeros(config.batch_size, config.max_dec_steps, self.n_tgt_vocab).cuda(), requires_grad=True)
		vocab_dist_ = initial_dist_.clone()

		if extra_zeros is not None:
			_, n_added = extra_zeros.size()
			#print(vocab_dist_.size(), extra_zeros.size())
			extra_zeros = extra_zeros.repeat(1, config.max_dec_steps).reshape(config.batch_size, config.max_dec_steps, n_added)
			#print(vocab_dist_.size(), extra_zeros.size())
			vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

		enc_batch_extend_vocab = enc_batch_extend_vocab.repeat(1, config.max_dec_steps).reshape(config.batch_size, config.max_dec_steps, config.max_article_len)
		#final_dist = vocab_dist_.scatter_add(2, enc_batch_extend_vocab, seq_logit)
		vocab_dist_.scatter_add_(2, enc_batch_extend_vocab, seq_logit)

		return vocab_dist_

		return final_dist.view(-1, final_dist.size(2))
