#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import h5py
import time
from optim_n2n import OptimN2N
from data import Dataset
from models_text import RNNVAE
import utils

import logger

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_file', default='data/ptb/ptb-train.hdf5')
parser.add_argument('--val_file', default='data/ptb/ptb-val.hdf5')
parser.add_argument('--test_file', default='data/ptb/ptb-test.hdf5')
parser.add_argument('--train_from', default='')

# Model options
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--enc_word_dim', default=256, type=int)
parser.add_argument('--enc_h_dim', default=256, type=int)
parser.add_argument('--enc_num_layers', default=1, type=int)
parser.add_argument('--dec_word_dim', default=256, type=int)
parser.add_argument('--dec_h_dim', default=256, type=int)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dec_dropout', default=0.5, type=float)
parser.add_argument('--model', default='vae', type=str, choices = ['vae', 'autoreg', 'savae', 'svi'])
parser.add_argument('--train_n2n', default=1, type=int)
parser.add_argument('--train_kl', default=1, type=int)

# Optimization options
parser.add_argument('--log_dir', default=None)
parser.add_argument('--checkpoint_dir', default='models/ptb')
parser.add_argument('--slurm', default=0, type=int)
parser.add_argument('--warmup', default=5, type=int)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--min_epochs', default=15, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--svi_steps', default=10, type=int)
parser.add_argument('--svi_lr1', default=1, type=float)
parser.add_argument('--svi_lr2', default=1, type=float)
parser.add_argument('--eps', default=1e-5, type=float)
parser.add_argument('--decay', default=0, type=int)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--svi_max_grad_norm', default=5, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--test', type=int, default=0)

parser.add_argument('--cycle', type=int, default=10)

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)
  test_data = Dataset(args.test_file)
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)
  logger.info('Train data: %d batches' % len(train_data))
  logger.info('Val data: %d batches' % len(val_data))
  logger.info('Test data: %d batches' % len(test_data))
  logger.info('Word vocab size: %d' % vocab_size)

  checkpoint_dir = args.checkpoint_dir
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  suffix = "%s_%s.pt" % (args.model, 'cyc')
  checkpoint_path = os.path.join(checkpoint_dir, suffix)

  if args.slurm == 0:
    cuda.set_device(args.gpu)
  if args.train_from == '':
    model = RNNVAE(vocab_size = vocab_size,
                   enc_word_dim = args.enc_word_dim,
                   enc_h_dim = args.enc_h_dim,
                   enc_num_layers = args.enc_num_layers,
                   dec_word_dim = args.dec_word_dim,
                   dec_h_dim = args.dec_h_dim,
                   dec_num_layers = args.dec_num_layers,
                   dec_dropout = args.dec_dropout,
                   latent_dim = args.latent_dim,
                   mode = args.model)
    for param in model.parameters():    
      param.data.uniform_(-0.1, 0.1)      
  else:
    logger.info('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']
      
  logger.info("model architecture")
  print(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  if args.warmup == 0:
    args.beta = 1.
  else:
    args.beta = 0.1
    
  criterion = nn.NLLLoss()
  model.cuda()
  criterion.cuda()
  model.train()

  def variational_loss(input, sents, model, z = None):
    mean, logvar = input
    z_samples = model._reparameterize(mean, logvar, z)
    preds = model._dec_forward(sents, z_samples)
    nll = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(preds.size(1))])
    kl = utils.kl_loss_diag(mean, logvar)
    return nll + args.beta*kl

  update_params = list(model.dec.parameters())
  meta_optimizer = OptimN2N(variational_loss, model, update_params, eps = args.eps, 
                            lr = [args.svi_lr1, args.svi_lr2],
                            iters = args.svi_steps, momentum = args.momentum,
                            acc_param_grads= args.train_n2n == 1,  
                            max_grad_norm = args.svi_max_grad_norm)
  if args.test == 1:
    args.beta = 1
    test_data = Dataset(args.test_file)    
    eval(test_data, model, meta_optimizer)
    exit()
    
  t = 0
  best_val_nll = 1e5
  best_epoch = 0
  val_stats = []
  epoch = 0
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    logger.info('Starting epoch %d' % epoch)
    train_nll_vae = 0.
    train_nll_autoreg = 0.
    train_kl_vae = 0.
    train_nll_svi = 0.
    train_kl_svi = 0.
    train_kl_init_final = 0.
    num_sents = 0
    num_words = 0
    b = 0
    
    tmp = float((epoch-1)%args.cycle)/args.cycle
    cur_lr = args.lr*0.5*(1+np.cos(tmp*np.pi))     
    for param_group in optimizer.param_groups:
      param_group['lr'] = cur_lr

    if (epoch-1) % args.cycle == 0:
      args.beta = 0.1
      logger.info('KL annealing restart')

    for i in np.random.permutation(len(train_data)):
      if args.warmup > 0:
        args.beta = min(1, args.beta + 1./(args.warmup*len(train_data)))
      
      sents, length, batch_size = train_data[i]
      if args.gpu >= 0:
        sents = sents.cuda()
      b += 1
      
      optimizer.zero_grad()
      if args.model == 'autoreg':
        preds = model._dec_forward(sents, None, True)
        nll_autoreg = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        train_nll_autoreg += nll_autoreg.data[0]*batch_size
        nll_autoreg.backward()
      elif args.model == 'svi':        
        mean_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
        logvar_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents,
                                                  b % args.print_every == 0)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
        preds = model._dec_forward(sents, z_samples)
        nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        train_nll_svi += nll_svi.data[0]*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        train_kl_svi += kl_svi.data[0]*batch_size      
        var_loss = nll_svi + args.beta*kl_svi          
        var_loss.backward(retain_graph = True)
      else:
        mean, logvar = model._enc_forward(sents)
        z_samples = model._reparameterize(mean, logvar)
        preds = model._dec_forward(sents, z_samples)
        nll_vae = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        train_nll_vae += nll_vae.data[0]*batch_size
        kl_vae = utils.kl_loss_diag(mean, logvar)
        train_kl_vae += kl_vae.data[0]*batch_size        
        if args.model == 'vae':
          vae_loss = nll_vae + args.beta*kl_vae          
          vae_loss.backward(retain_graph = True)
        if args.model == 'savae':
          var_params = torch.cat([mean, logvar], 1)        
          mean_svi = Variable(mean.data, requires_grad = True)
          logvar_svi = Variable(logvar.data, requires_grad = True)
          var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents,
                                                  b % args.print_every == 0)
          mean_svi_final, logvar_svi_final = var_params_svi
          z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
          preds = model._dec_forward(sents, z_samples)
          nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
          train_nll_svi += nll_svi.data[0]*batch_size
          kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
          train_kl_svi += kl_svi.data[0]*batch_size      
          var_loss = nll_svi + args.beta*kl_svi          
          var_loss.backward(retain_graph = True)
          if args.train_n2n == 0:
            if args.train_kl == 1:
              mean_final = mean_svi_final.detach()
              logvar_final = logvar_svi_final.detach()            
              kl_init_final = utils.kl_loss(mean, logvar, mean_final, logvar_final)
              train_kl_init_final += kl_init_final.data[0]*batch_size
              kl_init_final.backward(retain_graph = True)              
            else:
              vae_loss = nll_vae + args.beta*kl_vae
              var_param_grads = torch.autograd.grad(vae_loss, [mean, logvar], retain_graph=True)
              var_param_grads = torch.cat(var_param_grads, 1)
              var_params.backward(var_param_grads, retain_graph=True)              
          else:
            var_param_grads = meta_optimizer.backward([mean_svi_final.grad, logvar_svi_final.grad],
                                                      b % args.print_every == 0)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)        
      optimizer.step()
      num_sents += batch_size
      num_words += batch_size * length
      
      if b % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).data[0]**0.5
        logger.info('Iters: %d, Epoch: %d, Batch: %d/%d, LR: %.4f, TrainARNLL: %.4f, TrainARPPL: %.2f, TrainVAE_NLL: %.4f, TrainVAE_REC: %.4f, TrainVAE_KL: %.4f, TrainVAE_PPL: %.2f, TrainSVI_NLL: %.2f, TrainSVI_REC: %.2f, TrainSVI_KL: %.4f, TrainSVI_PPL: %.2f, KLInitFinal: %.2f, |Param|: %.4f, BestValPerf: %.2f, BestEpoch: %d, Beta: %.4f, Throughput: %.2f examples/sec' %
              (t, epoch, b+1, len(train_data), cur_lr, 
               train_nll_autoreg / num_sents, np.exp(train_nll_autoreg / num_words), 
               (train_nll_vae + train_kl_vae)/num_sents,
               train_nll_vae / num_sents, train_kl_vae / num_sents,  
               np.exp((train_nll_vae + train_kl_vae)/num_words),
               (train_nll_svi + train_kl_svi)/num_sents,
               train_nll_svi/num_sents, train_kl_svi/ num_sents,
               np.exp((train_nll_svi + train_kl_svi)/num_words), train_kl_init_final / num_sents,
               param_norm, best_val_nll, best_epoch, args.beta,
               num_sents / (time.time() - start_time)))
    
    epoch_train_time = time.time() - start_time
    logger.info('Time Elapsed: %.1fs' % epoch_train_time)   

    logger.info('--------------------------------')
    logger.info('Checking validation perf...')
    logger.record_tabular('Epoch', epoch)
    logger.record_tabular('Mode', 'Val')
    logger.record_tabular('LR', cur_lr)
    logger.record_tabular('Epoch Train Time', epoch_train_time)
    val_nll = eval(val_data, model, meta_optimizer)
    val_stats.append(val_nll)

    logger.info('--------------------------------')
    logger.info('Checking test perf...')
    logger.record_tabular('Epoch', epoch)
    logger.record_tabular('Mode', 'Test')
    logger.record_tabular('LR', cur_lr)
    logger.record_tabular('Epoch Train Time', epoch_train_time)
    test_nll = eval(test_data, model, meta_optimizer)

    if val_nll < best_val_nll:
      best_val_nll = val_nll
      best_epoch = epoch
      model.cpu()
      checkpoint = {
        'args': args.__dict__,
        'model': model,
        'val_stats': val_stats
      }
      logger.info('Save checkpoint to %s' % checkpoint_path)      
      torch.save(checkpoint, checkpoint_path)
      model.cuda()
    else:
      if epoch >= args.min_epochs:
        args.decay = 1
    # if args.decay == 1:
    #   args.lr = args.lr*0.5      
    #   for param_group in optimizer.param_groups:
    #     param_group['lr'] = args.lr
    #   if args.lr < 0.03:
    #     break
      
def eval(data, model, meta_optimizer):
    
  model.eval()
  criterion = nn.NLLLoss().cuda() 
  num_sents = 0
  num_words = 0
  total_nll_autoreg = 0.
  total_nll_vae = 0.
  total_kl_vae = 0.
  total_nll_svi = 0.
  total_kl_svi = 0.
  best_svi_loss = 0.
  for i in range(len(data)):
    sents, length, batch_size = data[i]
    num_words += batch_size*length
    num_sents += batch_size
    if args.gpu >= 0:
      sents = sents.cuda()
    if args.model == 'autoreg':
      preds = model._dec_forward(sents, None, True)
      nll_autoreg = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      total_nll_autoreg += nll_autoreg.data[0]*batch_size
    elif args.model == 'svi':
      mean_svi = Variable(0.1*torch.randn(batch_size, args.latent_dim).cuda(), requires_grad = True)
      logvar_svi = Variable(0.1*torch.randn(batch_size, args.latent_dim).cuda(), requires_grad = True)
      var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents)
      mean_svi_final, logvar_svi_final = var_params_svi
      z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
      preds = model._dec_forward(sents, z_samples)
      nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      total_nll_svi += nll_svi.data[0]*batch_size
      kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
      total_kl_svi += kl_svi.data[0]*batch_size
      mean, logvar = mean_svi_final, logvar_svi_final
    else:
      mean, logvar = model._enc_forward(sents)
      z_samples = model._reparameterize(mean, logvar)
      preds = model._dec_forward(sents, z_samples)
      nll_vae = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      total_nll_vae += nll_vae.data[0]*batch_size
      kl_vae = utils.kl_loss_diag(mean, logvar)
      total_kl_vae += kl_vae.data[0]*batch_size        
      if args.model == 'savae':
        mean_svi = Variable(mean.data, requires_grad = True)
        logvar_svi = Variable(logvar.data, requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
        preds = model._dec_forward(sents, z_samples)
        nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        total_nll_svi += nll_svi.data[0]*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        total_kl_svi += kl_svi.data[0]*batch_size      
        mean, logvar = mean_svi_final, logvar_svi_final

  nll_autoreg = total_nll_autoreg / num_sents
  ppl_autoreg = np.exp(total_nll_autoreg / num_words)
  nll_vae = (total_nll_vae + total_kl_vae)/num_sents
  rec_vae = total_nll_vae/ num_sents
  kl_vae = total_kl_vae / num_sents
  ppl_bound_vae = np.exp((total_nll_vae + total_kl_vae)/num_words)
  nll_svi = (total_nll_svi + total_kl_svi)/num_sents
  rec_svi = total_nll_svi/num_sents
  kl_svi = total_kl_svi/num_sents
  ppl_bound_svi = np.exp((total_nll_svi + total_kl_svi)/num_words)

  logger.record_tabular('AR NLL', nll_autoreg)
  logger.record_tabular('AR PPL', ppl_autoreg)
  logger.record_tabular('VAE NLL', nll_vae)
  logger.record_tabular('VAE REC', rec_vae)
  logger.record_tabular('VAE KL', kl_vae)
  logger.record_tabular('VAE PPL', ppl_bound_vae)
  logger.record_tabular('SVI NLL', nll_svi)
  logger.record_tabular('SVI REC', rec_svi)
  logger.record_tabular('SVI KL', kl_svi)
  logger.record_tabular('SVI PPL', ppl_bound_svi)
  logger.dump_tabular()
  logger.info('AR NLL: %.4f, AR PPL: %.4f, VAE NLL: %.4f, VAE REC: %.4f, VAE KL: %.4f, VAE PPL: %.4f, SVI NLL: %.4f, SVI REC: %.4f, SVI KL: %.4f, SVI PPL: %.4f' %
        (nll_autoreg, ppl_autoreg, nll_vae, rec_vae, kl_vae, ppl_bound_vae, nll_svi, rec_svi, kl_svi, ppl_bound_svi))
  model.train()
  if args.model == 'autoreg':
    return ppl_autoreg
  elif args.model == 'vae':
    return ppl_bound_vae
  elif args.model == 'savae' or args.model == 'svi':
    return ppl_bound_svi

if __name__ == '__main__':
  args = parser.parse_args()
  with logger.session(dir=args.log_dir, format_strs=['stdout', 'csv', 'log']):
    main(args)
