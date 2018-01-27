# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import os
from scipy.misc import toimage
from model import DAE, VAE, SCAN
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
parser.add_argument('--phase', default='scan', help='dae|vae|scan')

opt = parser.parse_args()
print(opt)

use_cuda = True
if not torch.cuda.is_available() or opt.cpu: use_cuda = False
if use_cuda: print('------CUDA USED------')
os.system('mkdir {0}'.format(opt.exp))
exp = opt.exp+'/'+opt.phase
os.system('mkdir {0}'.format(exp))


def train_scan(dae, vae, scan, data_manager, optimizer, begin_epoch=0, batch_size=16, training_epochs=1500, display_epoch=1, save_epoch=50):
  print("start training SCAN")
  step = 0
  scan.train()
  for epoch in range(begin_epoch, training_epochs):
    average_reconstr_loss = 0.0
    average_latent_loss0  = 0.0
    average_latent_loss1  = 0.0
    total_batch = int(IMAGE_CAPACITY / batch_size)
    
    for i in range(total_batch):
      batch_xs, batch_ys = data_manager.next_batch(batch_size, use_labels=True)
      data = Variable(torch.from_numpy(np.array(batch_xs)))
      target = Variable(torch.from_numpy(np.array(batch_ys)))
      if use_cuda:
        data, target = data.cuda(), target.cuda()
      data = torch.transpose(data,1,3)
      data = torch.transpose(data,2,3)

      optimizer.zero_grad()
      y_out, mu, logvar = scan(target)
      
      _, x_mu, x_logvar = vae(data)
      
      reconstr_loss, latent_loss0, latent_loss1 = scan.compute_loss(data, y_out, target, mu, logvar, x_mu, x_logvar)
      loss = reconstr_loss + latent_loss0 + latent_loss1
      loss.backward()
      optimizer.step()
      
      # Compute average loss
      average_reconstr_loss += reconstr_loss.data[0] / IMAGE_CAPACITY * batch_size
      average_latent_loss0  += latent_loss0.data[0]  / IMAGE_CAPACITY * batch_size
      average_latent_loss1  += latent_loss1.data[0]  / IMAGE_CAPACITY * batch_size

      step += 1
      if epoch % display_epoch == 0:
        print("Epoch:", '%04d' % (epoch+1),
          "reconstr=", "{:.3f}".format(average_reconstr_loss),
          "latent0=",  "{:.3f}".format(average_latent_loss0),
          "latent1=",  "{:.3f}".format(average_latent_loss1))

    if (epoch % save_epoch == 0) or (epoch == training_epochs-1):
      torch.save(scan.state_dict(), '{}/scan_epoch_{}.pth'.format(exp, epoch))

  
data_manager = DataManager()
data_manager.prepare()

dae = DAE()
vae = VAE()
scan = SCAN()
if use_cuda:
  dae.load_state_dict(torch.load('save/dae/dae_epoch_2999.pth'))
  vae.load_state_dict(torch.load('save/vae/vae_epoch_2999.pth'))
  scan.load_state_dict(torch.load('save/scan/scan_epoch_1499.pth'))
  dae, vae, scan = dae.cuda(), vae.cuda(), scan.cuda()
else:
  dae.load_state_dict(torch.load('save/dae/dae_epoch_2999.pth', map_location=lambda storage, loc: storage))
  vae.load_state_dict(torch.load('save/vae/vae_epoch_2999.pth', map_location=lambda storage, loc: storage))
  scan.load_state_dict(torch.load(exp+'/'+opt.load, map_location=lambda storage, loc: storage))



if opt.train:
  scan_optimizer = optim.Adam(scan.parameters(), lr=1e-4, eps=1e-8)
  train_scan(dae, vae, scan, data_manager, scan_optimizer)


