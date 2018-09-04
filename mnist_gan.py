import os
import sys
import argparse
import numpy as np
import torch
import torchvision

from torch import nn
from torch import optim
from torch.nn import functional as F

import ops
import utils
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--dim', default=64, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='latent space width')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--output', default=784, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)

    args = parser.parse_args()
    return args


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        self.linear1 = nn.Linear(128, 4*4*4*self.dim)
        self.conv1 = nn.ConvTranspose2d(4*self.dim, 2*self.dim, 5)
        self.conv2 = nn.ConvTranspose2d(2*self.dim, self.dim, 5)
        self.conv3 = nn.ConvTranspose2d(self.dim, 1, 8, stride=2)
        self.relu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print ('G in: ', x.shape)
        x = self.relu(self.linear1(x))
        x = x.view(-1, 4*self.dim, 4, 4)
        x = self.relu(self.conv1(x))
        x = x[:, :, :7, :7]
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.sigmoid(x)
        x = x.view(-1, 28*28)
        #print ('G out: ', x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Discriminator'
        self.conv1 = nn.Conv2d(1, self.dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2)
        self.relu = nn.ELU(inplace=True)
        self.linear1 = nn.Linear(4*4*4*self.dim, 1)
        self.ln1 = nn.LayerNorm([64, 14, 14])
        self.ln2 = nn.LayerNorm([128, 7, 7])
        self.ln3 = nn.LayerNorm([256, 4, 4])

    def forward(self, x):
        # print ('D in: ', x.shape)
        x = x.view(-1, 1, 28, 28)
        x = self.relu(self.ln1(self.conv1(x)))
        x = self.relu(self.ln2(self.conv2(x)))
        x = self.relu(self.ln3(self.conv3(x)))
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear1(x)
        x = x.view(-1)
        # print ('D out: ', x.shape)
        return x


def inf_gen(data_gen):
    while True:
        for images, targets in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets)

def train(args):
    
    torch.manual_seed(8734)
    
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    print (netG, netD)

    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    mnist_train, mnist_test = datagen.load_mnist(args)
    train = inf_gen(mnist_train)
    print ('saving reals')
    reals, _ = next(train)
    path = 'results/mnist/reals.png'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/Improved-WGAN/'+path
    utils.save_images(reals.detach().cpu().numpy(), path)
    
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1)
    
    print ('==> Begin Training')
    for iter in range(args.epochs):
        ops.batch_zero_grad([netG, netD])
        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(args.disc_iters):
            data, targets = next(train)
            data = data.view(args.batch_size, 28*28).cuda()
            netD.zero_grad()
            d_real = netD(data).mean()
            d_real.backward(mone, retain_graph=True)
            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            with torch.no_grad():
                fake = netG(noise)
            fake.requires_grad_(True)
            d_fake = netD(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one, retain_graph=True)
            gp = ops.grad_penalty_1dim(args, netD, data, fake)
            gp.backward()
            d_cost = d_fake - d_real + gp
            wasserstein_d = d_real - d_fake
            optimD.step()

        for p in netD.parameters():
            p.requires_grad=False
        netG.zero_grad()
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        fake = netG(noise)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        g_cost = -G
        optimG.step()
       
        if iter % 100 == 0:
            print('iter: ', iter, 'train D cost', d_cost.cpu().item())
            print('iter: ', iter, 'train G cost', g_cost.cpu().item())
        if iter % 300 == 0:
            val_d_costs = []
            for i, (data, target) in enumerate(mnist_test):
                data = data.cuda()
                d = netD(data)
                val_d_cost = -d.mean().item()
                val_d_costs.append(val_d_cost)
            utils.generate_image(args, iter, netG)

if __name__ == '__main__':

    args = load_args()
    train(args)
