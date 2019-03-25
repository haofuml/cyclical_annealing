#!/usr/bin/env python3
import numpy as np
import h5py
import torch
from torch.autograd import Variable

class Dataset(object):
  def __init__(self, h5_file):
    data = h5py.File(h5_file, 'r') #get text data
    self.sents = self._convert(data['source']).long()
    self.sent_lengths = self._convert(data['source_l']).long()
    self.batch_size = self._convert(data['batch_l']).long()
    self.batch_idx = self._convert(data['batch_idx']).long()
    self.vocab_size = data['vocab_size'][0]
    self.num_batches = self.batch_idx.size(0)

  def _convert(self, x):
    return torch.from_numpy(np.asarray(x))

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    assert(idx < self.num_batches and idx >= 0)
    start_idx = self.batch_idx[idx]
    end_idx = start_idx + self.batch_size[idx]
    length = self.sent_lengths[idx]
    sents = self.sents[start_idx:end_idx]
    batch_size = self.batch_size[idx]
    data_batch = [Variable(sents[:, :length]), length-1, batch_size]
    return data_batch
