# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import numpy as np
import os
from scipy.misc import toimage
import matplotlib.pyplot as plt

from model import DAE, VAE, SCAN
import utils
from data_manager import DataManager
from data_manager import IMAGE_CAPACITY, OP_AND, OP_IN_COMMON, OP_IGNORE
from torch.nn import functional as F
from torch import nn, optim
from torch.autograd import Variable

exp = 'save/vae'
plt.switch_backend('agg')
use_cuda = True
if not torch.cuda.is_available(): use_cuda = False
if use_cuda: print('------CUDA USED------')

train_dae_flag = False
train_vae_flag = True
def save_10_images(hsv_images, file_name):
  plt.figure()
  fig, axes = plt.subplots(1, 10, figsize=(10, 1),
                           subplot_kw={'xticks': [], 'yticks': []})
  fig.subplots_adjust(hspace=0.1, wspace=0.1)

  for ax,image in zip(axes.flat, hsv_images):
    hsv_image = image.reshape((80,80,3))
    rgb_image = utils.convert_hsv_to_rgb(hsv_image)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.imshow(rgb_image)

  plt.savefig(file_name, bbox_inches='tight')
  plt.close(fig)
  plt.close()

def disentangle_check(dae, vae, data_manager, save_original=False):
  """ Generate disentangled images with Beta VAE """
  hsv_image = data_manager.get_image(obj_color=0, wall_color=0, floor_color=0, obj_id=0)
  rgb_image = utils.convert_hsv_to_rgb(hsv_image)
  
  if save_original:
    utils.save_image(rgb_image, "original.png")

  # Caclulate latent mean and variance of given image.
  batch_xs = [hsv_image]
  data = Variable(torch.from_numpy(np.array(batch_xs)))
  if use_cuda:
    data = data.cuda()
  data = torch.transpose(data,1,3)
  data = torch.transpose(data,2,3)
  recon_batch_v, mu, logvar = vae(data)
  recon_batch = dae(recon_batch_v)
  var = logvar.exp().data[0]


  # Print variance
  zss_str = ""
  for i,zss in enumerate(var):
    str = "z{0}={1:.2f}".format(i,zss)
    zss_str += str + ", "
  print(zss_str)

  # Save disentangled images
  z_m = mu.data[0]
  n_z = 32

  if not os.path.exists("disentangle_img"):
    os.mkdir("disentangle_img")

  for target_z_index in range(n_z):
    z_mean2 = torch.zeros((10, n_z))
    
    for ri in range(10):
      # Change z mean value from -3.0 to +3.0
      value = -3.0 + (6.0 / 9.0) * ri

      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[ri][i] = value
        else:
          z_mean2[ri][i] = z_m[i]

         
    z_mean2 = Variable(z_mean2)
    if use_cuda: z_mean2 = z_mean2.cuda()
    generated_xs_v = vae.decode(z_mean2)
    generated_xs = dae(generated_xs_v)
    file_name = "disentangle_img/check_z{0}.png".format(target_z_index)
    generated_xs = torch.transpose(generated_xs,2,3)
    generated_xs = torch.transpose(generated_xs,1,3)
    if use_cuda: hsv_image = generated_xs.data.cpu().numpy()
    else: hsv_image = generated_xs.data.numpy()
    print(hsv_image[0].shape)
    save_10_images(hsv_image, file_name)
    

data_manager = DataManager()
data_manager.prepare()
vae = VAE()
dae = DAE()
if use_cuda:
  dae.load_state_dict(torch.load('save/dae/dae_epoch_2999.pth'))
  vae = vae.cuda()
  dae = dae.cuda()
  vae.load_state_dict(torch.load('save/vae/vae_epoch_2900.pth'))
else:
  dae.load_state_dict(torch.load('save/dae/dae_epoch_2999.pth', map_location=lambda storage, loc: storage))
  vae.load_state_dict(torch.load('save/vae/vae_epoch_2900.pth', map_location=lambda storage, loc: storage))

disentangle_check(dae, vae, data_manager)
