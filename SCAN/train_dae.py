# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import os
from scipy.misc import toimage
from model import DAE
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
parser.add_argument('--phase', default='dae', help='dae|vae|scan')

opt = parser.parse_args()
print(opt)

use_cuda = True
if not torch.cuda.is_available() or opt.cpu: use_cuda = False
if use_cuda: print('------CUDA USED------')
os.system('mkdir {0}'.format(opt.exp))
exp = opt.exp+'/'+opt.phase
os.system('mkdir {0}'.format(exp))

def train_dae(dae, data_manager, optimizer, batch_size=100, training_epochs=3000, display_epoch=1, save_epoch=50):

  print("start training DAE")
  step = 0
  dae.train()
  for epoch in range(training_epochs):
    average_loss = 0.0
    total_batch = int(IMAGE_CAPACITY / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of masked and orignal images
      batch_xs_masked, batch_xs = data_manager.next_masked_batch(batch_size)
      data, target = Variable(torch.from_numpy(np.array(batch_xs_masked))), Variable(torch.from_numpy(np.array(batch_xs)))
      if use_cuda:
        data, target = data.cuda(), target.cuda()
      data = torch.transpose(data,1,3)
      data = torch.transpose(data,2,3)
      target = torch.transpose(target,1,3)
      target = torch.transpose(target,2,3)

      optimizer.zero_grad()
      recon_batch = dae(data)
      loss = dae.compute_loss(target, recon_batch)
      loss.backward()
      # Compute average loss
      average_loss += loss.data[0] / IMAGE_CAPACITY * batch_size
      optimizer.step()

      step += 1
      
     # Display logs per epoch step
    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1), "loss=", "{}".format(average_loss))
    
    if epoch % 10 == 0:
      recon_batch = torch.transpose(recon_batch,2,3)
      recon_batch = torch.transpose(recon_batch,1,3)
      
      if use_cuda: hsv_image = recon_batch.data.cpu().numpy()
      else: hsv_image = recon_batch.data.numpy()
      print(hsv_image[0].shape)
      rgb_image = utils.convert_hsv_to_rgb(hsv_image[0])
      utils.save_image(rgb_image, "{}/reconstr_epoch_{}.png".format(exp, epoch))
      
    if epoch % 10 == 0:
      target = torch.transpose(target,2,3)
      target = torch.transpose(target,1,3)
      if use_cuda: hsv_image_t = target.data.cpu().numpy()
      else: hsv_image_t = target.data.numpy()
      rgb_image_t = utils.convert_hsv_to_rgb(hsv_image_t[0])
      utils.save_image(rgb_image_t, "{}/target_epoch_{}.png".format(exp, epoch))
    
    # Save to checkpoint
    if (epoch % save_epoch == 0) or (epoch == training_epochs-1):
      torch.save(dae.state_dict(), '{}/dae_epoch_{}.pth'.format(exp, epoch))


data_manager = DataManager()
data_manager.prepare()
dae = DAE()
if opt.load != '':
  print('loading {}'.format(opt.load))
  if use_cuda:
    dae.load_state_dict(torch.load())
  else:
    dae.load_state_dict(torch.load(exp+'/'+opt.load, map_location=lambda storage, loc: storage))

if use_cuda: dae = dae.cuda()

if opt.train:
  dae_optimizer = optim.Adam(dae.parameters(), lr=1e-4, eps=1e-8)
  train_dae(dae, data_manager, dae_optimizer)


