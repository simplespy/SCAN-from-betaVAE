# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import os
from scipy.misc import toimage
from model import DAE, VAE, SCAN, Recombinator
import utils
from data_manager import DataManager
from data_manager import IMAGE_CAPACITY
from torch.nn import functional as F
from torch import nn, optim
from torch.autograd import Variable
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='start training process')
parser.add_argument('--load', default='', help='load checkpoint')
parser.add_argument('--exp', default='save', help='Where to store samples and models')
parser.add_argument('--cpu', action='store_true', help='CPU only')
parser.add_argument('--phase', default='recomb', help='dae|vae|scan|recomb')

opt = parser.parse_args()
print(opt)

use_cuda = True
if not torch.cuda.is_available() or opt.cpu: use_cuda = False
if use_cuda: print('------CUDA USED------')
os.system('mkdir {0}'.format(opt.exp))
exp = opt.exp+'/'+opt.phase
os.system('mkdir {0}'.format(exp))


def train_recomb(dae, vae, scan, recomb, data_manager, optimizer, begin_epoch=0, batch_size=100, training_epochs=100, display_epoch=1, save_epoch=10):
  print("start training Recombinator")
  step = 0
  recomb.train()
  for epoch in range(begin_epoch, training_epochs):
    average_loss = 0.0
    total_batch = int(IMAGE_CAPACITY / batch_size)
    for i in range(total_batch):
      batch_ys0, batch_ys1, batch_ys, batch_xs, batch_hs = data_manager.get_op_training_batch(batch_size)
      xs = Variable(torch.from_numpy(np.array(batch_xs)))
      ys0 = Variable(torch.from_numpy(np.array(batch_ys0)))
      ys1 = Variable(torch.from_numpy(np.array(batch_ys1)))
      ys = Variable(torch.from_numpy(np.array(batch_ys)))
      hs = torch.FloatTensor(batch_size, 1, 3).zero_()
      for i in range(batch_size): hs[i, 0, batch_hs[i]] = 1
      op = Variable(hs)
      if use_cuda:
        xs, ys, ys0, ys1, op = xs.cuda(), ys.cuda(), ys0.cuda(), ys1.cuda(), op.cuda()
      xs = torch.transpose(xs,1,3)
      xs = torch.transpose(xs,2,3)

      optimizer.zero_grad()
      mu_0, logvar_0 = scan.encode(ys0)
      mu_1, logvar_1 = scan.encode(ys1)
      y_mu, y_logvar = scan.encode(ys)
      x_mu, x_logvar = vae.encode(xs)

      r_z, r_mu, r_logvar = recomb(mu_0, logvar_0, mu_1, logvar_1, op)
      y_out = scan.decode(r_z)
      image_loss, symbol_loss = recomb.compute_loss(r_mu, r_logvar, x_mu, x_logvar, y_mu, y_logvar)
      loss = symbol_loss
      loss.backward()
      optimizer.step()
      average_loss += loss.data[0] / IMAGE_CAPACITY * batch_size
      step += 1

    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1), "loss=", "{}".format(average_loss))

    if (epoch % save_epoch == 0) or (epoch == training_epochs-1):
      torch.save(recomb.state_dict(), '{}/recomb_epoch_{}.pth'.format(exp, epoch))


  
data_manager = DataManager()
data_manager.prepare()

dae = DAE()
vae = VAE()
scan = SCAN()
recomb = Recombinator()

if use_cuda:
  dae.load_state_dict(torch.load('save/dae/dae_epoch_2999.pth'))
  vae.load_state_dict(torch.load('save/vae/vae_epoch_2999.pth'))
  scan.load_state_dict(torch.load('save/scan/scan_epoch_1499.pth'))
  dae, vae, scan, recomb = dae.cuda(), vae.cuda(), scan.cuda(), recomb.cuda()
else:
  dae.load_state_dict(torch.load('save/dae/dae_epoch_2999.pth', map_location=lambda storage, loc: storage))
  vae.load_state_dict(torch.load('save/vae/vae_epoch_2999.pth', map_location=lambda storage, loc: storage))
  scan.load_state_dict(torch.load('save/scan/scan_epoch_1499.pth', map_location=lambda storage, loc: storage))
  recomb.load_state_dict(torch.load(exp+'/'+opt.load, map_location=lambda storage, loc: storage))



if opt.train:
  recomb_optimizer = optim.Adam(recomb.parameters(), lr=1e-3, eps=1e-8)
  train_recomb(dae, vae, scan, recomb, data_manager, recomb_optimizer)


