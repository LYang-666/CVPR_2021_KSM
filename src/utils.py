"""Contains a bunch of utility functions."""

import numpy as np
from torchvision import models
import torch.nn.functional as F

import torch
from math import cos, pi

def step_lr_cos(epoch, base_lr, optimizer, iteration, num_iter):
    """Handles step decay of learning rate."""
    current_iter = iteration + epoch * num_iter
    max_iter = 30 * num_iter
    new_lr = base_lr * (1 + cos(pi * current_iter/max_iter)) * 0.5

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # print('Set lr to ', new_lr)
    return optimizer, new_lr

def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    new_lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Set lr to ', new_lr)
    return optimizer


def set_dataset_paths(args):
    """Set default train and test path if not provided as input."""
    if not args.train_path:
        args.train_path = '/dataset_nvme/imagenet_to_sketch/%s/train' % (args.dataset)
    if not args.test_path:
        if args.dataset == 'imagenet' or args.dataset == 'places':
            args.test_path = '../data/%s/val' % (args.dataset)
        else:
            args.test_path = '/dataset_nvme/imagenet_to_sketch/%s/test' % (args.dataset)


def get_network(network, depth, dataset, use_bn=True):
    if network == 'vgg':
        print('Use batch norm is: %s' % use_bn)
        return models.vgg16(pretrained=True)
    elif network == 'resnet':
        return models.resnet50(pretrained=True)
    else:
        raise NotImplementedError('Network unsupported ' + network)

def filter_weights(weights, mask):
    w = weights.view(-1).tolist()
    m = mask.view(-1).tolist()
    res = []
    for idx in range(len(m)):
        if m[idx] > 0.5:
            res.append(w[idx])
    return res


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        return

    def update(self, val, num):
        self.sum = self.sum + val * num
        self.n = self.n + num

    @property
    def avg(self):
        return self.sum / self.n


def classification_accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()




