import os
import sys
import time
import torch
import glob
import itertools
import numpy as np

from scipy.misc import imsave
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
from torchvision.utils import save_image


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad = grad
    return z


def sample_z_like(shape, scale=1., grad=True):
    return torch.randn(*shape, requires_grad=grad).cuda()


def save_model(args, model, optim):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'best_acc': args.best_acc,
        'best_loss': args.best_loss
        }, path)


def load_model(args, model, optim):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    acc = ckpt['best_acc']
    loss = ckpt['best_loss']
    return model, optim, (acc, loss)


def get_net_only(model):
    net_dict = {
            'state_dict': model.state_dict(),
    }
    return net_dict


def load_net_only(model, d):
    model.load_state_dict(d['state_dict'])
    return model


def generate_image(args, iter, netG):
    with torch.no_grad():
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        samples = netG(noise)
    if samples.dim() < 4:
        channels = 1
        out_size = int(np.sqrt(args.output))
        samples = samples.view(-1, channels, out_size, out_size)
    else:
        channels = samples.shape[1]
        out_size = int(np.sqrt(args.output//3))
        samples = samples.view(-1, channels, out_size, out_size)
       	samples = samples.mul(0.5).add(0.5) 
    path = 'results/samples_{}.png'.format(iter)
    if not os.path.exists('results'):
        os.makedirs('results')
    print ('saving sample: ', path)
    save_image(samples, path)
