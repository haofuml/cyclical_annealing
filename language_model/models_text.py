#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
                   
class RNNVAE(nn.Module):
  def __init__(self, vocab_size=10000,
               enc_word_dim = 512,
               enc_h_dim = 1024,
               enc_num_layers = 1,
               dec_word_dim = 512,
               dec_h_dim = 1024,
               dec_num_layers = 1,
               dec_dropout = 0.5,
               latent_dim=32,
               mode='savae'):
    super(RNNVAE, self).__init__()
    self.enc_h_dim = enc_h_dim
    self.enc_num_layers = enc_num_layers
    self.dec_h_dim =dec_h_dim
    self.dec_num_layers = dec_num_layers
    
    if mode == 'savae' or mode == 'vae':
      self.enc_word_vecs = nn.Embedding(vocab_size, enc_word_dim)
      self.latent_linear_mean = nn.Linear(enc_h_dim, latent_dim)
      self.latent_linear_logvar = nn.Linear(enc_h_dim, latent_dim)
      self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers = enc_num_layers,
                             batch_first = True)
      self.enc = nn.ModuleList([self.enc_word_vecs, self.enc_rnn,
                                self.latent_linear_mean, self.latent_linear_logvar])
    elif mode == 'autoreg':
      latent_dim = 0
      
    self.dec_word_vecs = nn.Embedding(vocab_size, dec_word_dim)
    dec_input_size = dec_word_dim
    dec_input_size += latent_dim
    self.dec_rnn = nn.LSTM(dec_input_size, dec_h_dim, num_layers = dec_num_layers,
                           batch_first = True)      
    self.dec_linear = nn.Sequential(*[nn.Dropout(dec_dropout),                                      
                                      nn.Linear(dec_h_dim, vocab_size),
                                      nn.LogSoftmax()])      
    self.dropout = nn.Dropout(dec_dropout)
    self.dec = nn.ModuleList([self.dec_word_vecs, self.dec_rnn, self.dec_linear])
    if latent_dim > 0:
      self.latent_hidden_linear = nn.Linear(latent_dim, dec_h_dim)
      self.dec.append(self.latent_hidden_linear)
    
  def _enc_forward(self, sent):
    word_vecs = self.enc_word_vecs(sent)
    h0 = Variable(torch.zeros(self.enc_num_layers, word_vecs.size(0),
                              self.enc_h_dim).type_as(word_vecs.data))
    c0 = Variable(torch.zeros(self.enc_num_layers, word_vecs.size(0),
                              self.enc_h_dim).type_as(word_vecs.data))
    enc_h_states, _ = self.enc_rnn(word_vecs, (h0, c0))
    enc_h_states_last = enc_h_states[:, -1]
    mean = self.latent_linear_mean(enc_h_states_last)
    logvar = self.latent_linear_logvar(enc_h_states_last)
    return mean, logvar
  
  def _reparameterize(self, mean, logvar, z = None):
    std = logvar.mul(0.5).exp()    
    if z is None:
      z = Variable(torch.cuda.FloatTensor(std.size()).normal_(0, 1))
    return z.mul(std) + mean
  
  def _dec_forward(self, sent, q_z, init_h = True):
    self.word_vecs = self.dropout(self.dec_word_vecs(sent[:, :-1]))
    if init_h:
      self.h0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0),
                                     self.dec_h_dim).type_as(self.word_vecs.data), requires_grad = False)
      self.c0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0),
                                     self.dec_h_dim).type_as(self.word_vecs.data), requires_grad = False)
    else:
      self.h0.data.zero_()
      self.c0.data.zero_()
    
    if q_z is not None:
      q_z_expand = q_z.unsqueeze(1).expand(self.word_vecs.size(0),
                                           self.word_vecs.size(1), q_z.size(1))
      dec_input = torch.cat([self.word_vecs, q_z_expand], 2)
    else:
      dec_input = self.word_vecs
    if q_z is not None:
      self.h0[-1] = self.latent_hidden_linear(q_z)
    memory, _ = self.dec_rnn(dec_input, (self.h0, self.c0))
    dec_linear_input = memory.contiguous()
    preds = self.dec_linear(dec_linear_input.view(
      self.word_vecs.size(0)*self.word_vecs.size(1), -1)).view(
        self.word_vecs.size(0), self.word_vecs.size(1), -1)
    return preds

  
