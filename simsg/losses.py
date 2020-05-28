#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


def get_gan_losses(gan_type):
  """
  Returns the generator and discriminator loss for a particular GAN type.

  The returned functions have the following API:
  loss_g = g_loss(scores_fake)
  loss_d = d_loss(scores_real, scores_fake)
  """
  if gan_type == 'gan':
    return gan_g_loss, gan_d_loss
  elif gan_type == 'wgan':
    return wgan_g_loss, wgan_d_loss
  elif gan_type == 'lsgan':
    return lsgan_g_loss, lsgan_d_loss
  else:
    raise ValueError('Unrecognized GAN type "%s"' % gan_type)

def gan_percept_loss(real, fake):

  '''
  Inputs:
  - real: discriminator feat maps for every layer, when x=real image
  - fake: discriminator feat maps for every layer, when x=pred image
  Returns:
    perceptual loss in all discriminator layers
  '''

  loss = 0

  for i in range(len(real)):
    loss += (real[i] - fake[i]).abs().mean()

  return loss / len(real)


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def _make_targets(x, y):
  """
  Inputs:
  - x: PyTorch Tensor
  - y: Python scalar

  Outputs:
  - out: PyTorch Variable with same shape and dtype as x, but filled with y
  """
  return torch.full_like(x, y)


def gan_g_loss(scores_fake):
  """
  Input:
  - scores_fake: Tensor of shape (N,) containing scores for fake samples

  Output:
  - loss: Variable of shape (,) giving GAN generator loss
  """
  if scores_fake.dim() > 1:
    scores_fake = scores_fake.view(-1)
  y_fake = _make_targets(scores_fake, 1)
  return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
  """
  Input:
  - scores_real: Tensor of shape (N,) giving scores for real samples
  - scores_fake: Tensor of shape (N,) giving scores for fake samples

  Output:
  - loss: Tensor of shape (,) giving GAN discriminator loss
  """
  assert scores_real.size() == scores_fake.size()
  if scores_real.dim() > 1:
    scores_real = scores_real.view(-1)
    scores_fake = scores_fake.view(-1)
  y_real = _make_targets(scores_real, 1)
  y_fake = _make_targets(scores_fake, 0)
  loss_real = bce_loss(scores_real, y_real)
  loss_fake = bce_loss(scores_fake, y_fake)
  return loss_real + loss_fake


def wgan_g_loss(scores_fake):
  """
  Input:
  - scores_fake: Tensor of shape (N,) containing scores for fake samples

  Output:
  - loss: Tensor of shape (,) giving WGAN generator loss
  """
  return -scores_fake.mean()


def wgan_d_loss(scores_real, scores_fake):
  """
  Input:
  - scores_real: Tensor of shape (N,) giving scores for real samples
  - scores_fake: Tensor of shape (N,) giving scores for fake samples

  Output:
  - loss: Tensor of shape (,) giving WGAN discriminator loss
  """
  return scores_fake.mean() - scores_real.mean()


def lsgan_g_loss(scores_fake):
  if scores_fake.dim() > 1:
    scores_fake = scores_fake.view(-1)
  y_fake = _make_targets(scores_fake, 1)
  return F.mse_loss(scores_fake.sigmoid(), y_fake)


def lsgan_d_loss(scores_real, scores_fake):
  assert scores_real.size() == scores_fake.size()
  if scores_real.dim() > 1:
    scores_real = scores_real.view(-1)
    scores_fake = scores_fake.view(-1)
  y_real = _make_targets(scores_real, 1)
  y_fake = _make_targets(scores_fake, 0)
  loss_real = F.mse_loss(scores_real.sigmoid(), y_real)
  loss_fake = F.mse_loss(scores_fake.sigmoid(), y_fake)
  return loss_real + loss_fake


def gradient_penalty(x_real, x_fake, f, gamma=1.0):
  N = x_real.size(0)
  device, dtype = x_real.device, x_real.dtype
  eps = torch.randn(N, 1, 1, 1, device=device, dtype=dtype)
  x_hat = eps * x_real + (1 - eps) * x_fake
  x_hat_score = f(x_hat)
  if x_hat_score.dim() > 1:
    x_hat_score = x_hat_score.view(x_hat_score.size(0), -1).mean(dim=1)
  x_hat_score = x_hat_score.sum()
  grad_x_hat, = torch.autograd.grad(x_hat_score, x_hat, create_graph=True)
  grad_x_hat_norm = grad_x_hat.contiguous().view(N, -1).norm(p=2, dim=1)
  gp_loss = (grad_x_hat_norm - gamma).pow(2).div(gamma * gamma).mean()
  return gp_loss

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
from simsg.SPADE.architectures import VGG19


#                            SPADE losses!                      #
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


