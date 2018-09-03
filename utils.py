import os
import sys
import time
import torch
import natsort
import datagen
import scipy.misc
import numpy as np
import itertools
import cv2
import numpy as np

from glob import glob
from scipy.misc import imsave
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N


param_dir = './params/sampled/mnist/test1/'
model_dir = 'models/HyperGAN/'


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


def save_clf(args, Z, acc):
    import models.mnist_clf as models
    model = models.Small2().cuda()
    state = model.state_dict()
    layers = zip(args.stat['layer_names'], Z)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        assert state[name].equal(loader) == False
        model.load_state_dict(state)
    path = 'exp_models/hypermnist_clf_{}.pt'.format(acc)
    print ('saving hypernet to {}'.format(path))
    torch.save({'state_dict': model.state_dict()}, path)


def save_hypernet(args, models, acc):
    netE, W1, W2, W3 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            }
    path = 'exp_models/hypermnist_{}.pt'.format(acc)
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


""" hard coded for mnist experiment dont use generally """
def load_hypernet(path, args=None):
    if args is None:
        args = load_default_args()
    netE = hyper.Encoder_small(args).cuda()
    W1 = hyper.GeneratorW1_small(args).cuda()
    W2 = hyper.GeneratorW2_small(args).cuda()
    W3 = hyper.GeneratorW3_small(args).cuda()
    print ('loading hypernet from {}'.format(path))
    d = torch.load(path)
    netE = load_net_only(netE, d['E'])
    W1 = load_net_only(W1, d['W1'])
    W2 = load_net_only(W2, d['W2'])
    W3 = load_net_only(W3, d['W3'])
    return (netE, W1, W2, W3)


def load_default_args():
    parser = argparse.ArgumentParser(description='default hyper-args')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=300, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='small2', type=str)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--boost', default=10, type=int)
    args = parser.parse_args()
    return args


def dataset_iterator(args, id):
    train_gen, dev_gen = datagen.load(args, id)
    return (train_gen, dev_gen)


def inf_train_gen(train_gen):
    if type(train_gen) is list:
        while True:
            for (p1) in (train_gen[0](), train_gen[1]()):
                yield (p1)
    else:
        while True:
            for params in train_gen():
                yield params


def load_params(flat=True):
    paths = glob(param_dir+'/*.npy')
    paths = natsort.natsorted(paths)
    s = np.load(paths[0]).shape
    # print (s)
    params = np.zeros((len(paths), *s))
    # print (params.shape)
    for i in range(len(paths)):
        params[i] = np.load(paths[i])

    if flat is True:
        res = params.flatten()
        params = res
    return res


def save_samples(args, samples, iter, path):
    # lets view the first filter
    filters = samples[:, 0, :, :]
    filters = filters.unsqueeze(3)
    grid_img = grid(16, 8, filters, margin=2)
    im_path = 'plots/{}/{}/filters/{}.png'.format(args.dataset, args.model, iter)
    cv2.imwrite(im_path, grid_img)
    return


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def grid(w, h, imgs, margin):
    n = w*h
    img_h, img_w, img_c = imgs[0].shape
    m_x = 0
    m_y = 0
    if margin is not None:
        m_x = int(margin)
        m_y = m_x
    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
        img_w * w + m_x * (w - 1),
        img_c),
        np.uint8)
    imgmatrix.fill(255)    

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img
    return imgmatrix


def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, n_samples//rows
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])),
            int(np.sqrt(X.shape[1]))))
    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)


def generate_image(args, iter, netG):
    #noise = sample_d(dist, args.batch_size)
    noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
    samples = netG(noise)
    samples = samples.view(args.batch_size, 28, 28)
    samples = samples.cpu().data.numpy()
    print ('saving sample: results/mnist/samples_{}.png'.format(iter))
    save_images(samples, 'results/mnist/samples_{}.png'.format(iter))
