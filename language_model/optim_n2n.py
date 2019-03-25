#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil

import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OptimN2N:
  def __init__(self, loss_fn, model, model_update_params,
               lr=[1],
               iters=20,
               acc_param_grads=True,
               max_grad_norm = 0,
               eps = 0.00001,
               momentum=0.5):
       
    self.iters = iters
    self.lr = lr
    self.loss_fn = loss_fn
    self.eps = eps
    self.max_grad_norm = max_grad_norm
    self.model = model
    self.momentum = momentum
    self.acc_param_grads = acc_param_grads
    if self.acc_param_grads:
      self.params = model_update_params
      self.param_grads = [torch.zeros([self.iters] + list(p.size())).type_as(p.data)
                          for p in self.params]
    
  def forward(self, input, y, verbose=False):
    self.seeds = np.random.randint(3435, size=self.iters)
    return self.forward_mom(input, y, verbose)

  def backward(self, grad_output, verbose=False):
    grads = self.backward_mom(grad_output, verbose)
    return grads                             

      
  def grad_norm(self, g_list, g_norm_list):
    for g, g_norm in zip(g_list, g_norm_list):
      g_norm2 = (g**2).sum(1)**0.5
      g.div_(g_norm2.unsqueeze(1).expand_as(g)).mul_(g_norm.unsqueeze(1).expand_as(g))
      
  def clip_grad_norm(self, parameters, max_norm, norm_type=2):
    if len(parameters) > 0:
      max_norm = float(max_norm)
      norm_type = float(norm_type)
      if norm_type == float('inf'):
        total_norm = max(p.abs().max() for p in parameters)
      else:
        total_norm = 0
        for p in parameters:
          param_norm = p.norm(norm_type)
          total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
          for p in parameters:
            p.mul_(clip_coef)
            
  def forward_mom(self, input, y, verbose=False):
    self.y = y
    self.input_grads = [torch.zeros([self.iters] + list(x.size())).type_as(x.data) for x in input]
    self.mom_params = [torch.zeros(x.size()).type_as(x) for x in self.input_grads]    
    self.input_cache = [torch.zeros(x.size()).type_as(x) for x in self.input_grads]
    self.all_z = []    
    if self.acc_param_grads:
      for p in self.param_grads:
        p.zero_()
    for k in range(self.iters):
      self.all_z.append(Variable(torch.cuda.FloatTensor(input[0].size()).normal_(0, 1)))
      torch.manual_seed(int(self.seeds[k]))
      loss = self.loss_fn(input, self.y, self.model, self.all_z[k])
        
      if self.acc_param_grads:
        all_input_params = input + self.params
      else:
        all_input_params = input        
      all_grads_k = torch.autograd.grad(loss, all_input_params, retain_graph = True)
      input_grad_k = all_grads_k[:len(input)]
      param_grads_k = all_grads_k[len(input):]
      
      if self.max_grad_norm > 0:        
        self.clip_grad_norm([input_grad_k[0].data], self.max_grad_norm)
        self.clip_grad_norm([input_grad_k[1].data], self.max_grad_norm)
        
      if self.acc_param_grads:
        for i, p in enumerate(param_grads_k):
          self.param_grads[i][k].copy_(p.data)
      for i, x_grad_k in enumerate(input_grad_k):
        self.input_cache[i][k].copy_(input[i].data)        
        self.input_grads[i][k].copy_(x_grad_k.data)        
        
      for i in range(len(self.mom_params)):
        if k == 0:
          self.mom_params[i][k] = -input_grad_k[i].data
        else:
          self.mom_params[i][k] = self.mom_params[i][k-1]*self.momentum -input_grad_k[i].data
      lr_k_list = [lr for lr in self.lr]
      input = [Variable(x.data + lr_k * p[k], requires_grad=True) for x, p, lr_k in
               zip(input, self.mom_params, lr_k_list)]      
      if verbose:
        print('mom', k, loss.data[0])
    return input

  def backward_mom(self, grad_output, verbose=False):
    input_kp1_grad = [g.data for g in grad_output]
    p_kp1_grad = [torch.zeros(x.size()).type_as(x) for x in input_kp1_grad]
    rev_iters = self.iters
    for k in reversed(range(rev_iters)):
      lr_k_list = [lr for lr in self.lr]
      input_k_grad = input_kp1_grad
      p_kp1_grad = [p + lr_k*x for p, x, lr_k in zip(p_kp1_grad, input_kp1_grad, lr_k_list)]
      input_k_rv = []
      input_H_xx_v = []
      r = self.eps
      for i in range(len(p_kp1_grad)):
        v = p_kp1_grad[i]
        x_k = self.input_cache[i][k]
        x_k_rv = Variable((x_k + r*v).type_as(x_k), requires_grad = True)
        input_k_rv.append(x_k_rv)
      if self.acc_param_grads:
        all_input_params = input_k_rv + self.params
      else:
        all_input_params = input_k_rv
      torch.manual_seed(int(self.seeds[k]))
      loss = self.loss_fn(input_k_rv, self.y, self.model, self.all_z[k])        
      all_grads_rv_k = torch.autograd.grad(loss, all_input_params, retain_graph=True)
      
      if self.max_grad_norm > 0:
        self.clip_grad_norm([g.data for g in all_grads_rv_k], self.max_grad_norm)
        
      input_grads_rv_k = all_grads_rv_k[:len(input_k_rv)]
      param_grads_rv_k = all_grads_rv_k[len(input_k_rv):]      
      
      if self.acc_param_grads:
        H_wx_v_list = []
        for i, p_grad_rv_k in enumerate(param_grads_rv_k):
          H_wx_v = (p_grad_rv_k.data - self.param_grads[i][k]) / r
          H_wx_v_list.append(H_wx_v)
          if self.params[i].grad is None:
            self.params[i].grad = Variable(torch.zeros(self.params[i].size()).type_as(
              self.params[i].data))
        if self.max_grad_norm > 0:
          self.clip_grad_norm(H_wx_v_list, self.max_grad_norm)                                
        for i in range(len(self.params)):
          self.params[i].grad.data += -H_wx_v_list[i]           
      for i, x_k_rv_grad in enumerate(input_grads_rv_k):
        H_xx_v = (x_k_rv_grad.data - self.input_grads[i][k])/r
        input_H_xx_v.append(H_xx_v)
      input_kp1_grad = [x_kp1_grad - H_xx_v
                        for (x_kp1_grad, H_xx_v) in zip(input_kp1_grad, input_H_xx_v)]
      if self.max_grad_norm > 0:
        self.clip_grad_norm(input_kp1_grad, self.max_grad_norm)                                
      p_kp1_grad = [p.mul_(self.momentum) for p in p_kp1_grad]
      if verbose:
        print('mom', k, input_kp1_grad[0][0].norm())        
    return input_kp1_grad

