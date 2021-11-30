"""Main entry point for doing all stuff."""
from __future__ import division, print_function

import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm
import time
import dataset
import networks as net
import utils as utils
import matplotlib.pyplot as plt

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--arch',
                   choices=['vgg16', 'vgg16bn', 'resnet50',
                            'densenet121', 'resnet50_diff'],
                   help='Architectures')
FLAGS.add_argument('--source', type=str, default='',
                   help='Location of the init file for resnet50_diff')
FLAGS.add_argument('--finetune_layers',
                   choices=['all', 'fc', 'classifier'], default='all',
                   help='Which layers to finetune, fc only works with vgg')
FLAGS.add_argument('--mode',
                   choices=['finetune', 'masking', 'eval', 'check'],
                   help='Run mode')
FLAGS.add_argument('--num_outputs', type=int, default=-1,
                   help='Num outputs for dataset')
# Optimization options.
FLAGS.add_argument('--lr', type=float,
                   help='Learning rate for parameters, used for baselines')
FLAGS.add_argument('--lr_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--lr_mask', type=float,
                   help='Learning rate for mask')
FLAGS.add_argument('--lr_mask_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--mask_adam', action='store_true', default=False,
                   help='Use adam instead of sgdm for masks')
FLAGS.add_argument('--lr_classifier', type=float,
                   help='Learning rate for classifier')
FLAGS.add_argument('--finetune_epochs', type=int,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--batch_size', type=int, default=32,
                   help='Batch size')
FLAGS.add_argument('--weight_decay', type=float, default=0.0,
                   help='Weight decay')
FLAGS.add_argument('--train_bn', action='store_true', default=True,
                   help='train batch norm or not')
# Masking options.

FLAGS.add_argument('--mask_scale', type=float, default=1e-2,
                   help='Mask initialization scaling')
# Paths.
FLAGS.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--save_prefix', type=str, default='./checkpoints/debug',
                   help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='',
                   help='Location to save model')
# Other.
FLAGS.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')
FLAGS.add_argument('--no_mask', action='store_true', default=False,
                   help='Used for running baselines, does not use any masking')


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model):
        self.args = args
        self.cuda = args.cuda
        self.model = model

        # Set up data loader, criterion, and pruner.
        if 'cropped' in args.train_path:
            train_loader = dataset.train_loader_cropped
            test_loader = dataset.test_loader_cropped
        else:
            train_loader = dataset.train_loader
            test_loader = dataset.test_loader
        self.train_data_loader = train_loader(
            args.train_path, args.batch_size, pin_memory=args.cuda)
        self.test_data_loader = test_loader(
            args.test_path, args.batch_size, pin_memory=args.cuda)
        self.criterion = nn.CrossEntropyLoss()

    def eval(self,epoch_idx):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None
        val_loss = utils.Metric('val_loss')
        val_accuracy = utils.Metric('val_accuracy')
        cap_loss = utils.Metric('cap_loss')
        print('Performing eval...')
        with tqdm(total=len(self.test_data_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx),
                  ascii=True) as t:

            with torch.no_grad():
                for data, label in self.test_data_loader:
                    if self.cuda:
                        data = data.cuda()
                        label = label.cuda()

                    output = self.model(data)
                    loss = self.criterion(output, label)

                    num = data.size(0)
                    val_loss.update(loss, num)
                    val_accuracy.update(utils.classification_accuracy(output, label), num)
                    
                    t.set_postfix({'loss': val_loss.avg.item(), 
                            'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item())})
                    t.update(1)
                    
                    # Init error meter.
                    if error_meter is None:
                        topk = [1]
                        if output.size(1) > 5:
                            topk.append(5)
                        error_meter = tnt.meter.ClassErrorMeter(topk=topk)
                    error_meter.add(output.data, label)


        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))

        return errors, val_loss.avg.item()

    def do_epoch(self, epoch_idx, optimizer, savename):
        """Trains model for one epoch."""
        train_loss = utils.Metric('train_loss')
        train_accuracy = utils.Metric('train_accuracy')


        with tqdm(total=len(self.train_data_loader),
            desc='Train Ep. #{}: '.format(epoch_idx),
            disable=False,
            ascii=True) as t:

            for i, (batch, label) in enumerate(self.train_data_loader):
                if self.cuda:
                    batch = batch.cuda()
                    label = label.cuda()

                # using cosine lr decay in here
                optimizer.update_lr_cos(epoch_idx, i, len(self.train_data_loader))

                # Set grads to 0.
                self.model.zero_grad()

                # Do forward-backward.
                output = self.model(batch)
                num = batch.size(0)

                train_accuracy.update(utils.classification_accuracy(output, label), num)
                loss = self.criterion(output, label)

                train_loss.update(loss, num)
                loss.backward()
                optimizer.step()

                t.set_postfix({'loss': train_loss.avg.item(), 
                        'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item())})
                t.update(1)

        # calculate sparsity
        total_mask = 0
        total_mask_one = 0
        for idx, module in enumerate(self.model.shared.modules()):
            if 'GroupWise' in str(type(module)):
                num_mask = module.mask_real.numel()
                # num_one = module.bin_mask.data.eq(1).sum().item()
                num_one = module.total_non_zeros.item()
                total_mask += num_mask
                total_mask_one += num_one
                print('{},{}, -- sparsity:{:2.4f}'.format(idx, module.weight.size(), 1-num_one/num_mask))
        print('total soft factor ratio:{:2.4f}'.format(1-total_mask_one/total_mask))



        return train_loss.avg.item(), 100. * train_accuracy.avg.item(), 1-total_mask_one/total_mask

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'model': self.model.state_dict(),
        }

        # Save to file.
        torch.save(ckpt, savename)

    def train(self, epochs, optimizer, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []
        sparsity_history = []
        if self.args.cuda:
            self.model = self.model.cuda()
                 
        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            # optimizer.update_lr(epoch_idx)
            if self.args.train_bn:
                self.model.train()
            else:
                self.model.train_nobn()
            train_los, train_acc, m_sparsity = self.do_epoch(epoch_idx, optimizer,savename)
            
            sparsity_history.append(m_sparsity)
            errors, val_los = self.eval(epoch_idx)
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.
            print('-- train_los :{:2.4f} -- val_los:{:2.4f} -- train_acc:{:2.4f} -- test_acc:{:2.4f}'.format(train_los, val_los, train_acc, accuracy))

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'sparsity_history': sparsity_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        # Make sure masking didn't change any weights.

        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, args):
        self.optimizers = []
        self.lrs = []
        self.args = args

    def add(self, optimizer, lr):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr_cos(self, epoch_idx, iteration, num_iter):
         """Update learning rate of every optimizer."""
         for optimizer, init_lr in zip(self.optimizers, self.lrs):
             optimizer = utils.step_lr_cos(
                 epoch_idx, init_lr, optimizer, iteration, num_iter)

def main():
    """Do stuff."""
    args = FLAGS.parse_args()

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    # Load the required model.
    if args.arch == 'vgg16':
        model = net.ModifiedVGG16(mask_init=args.mask_init,
                                  mask_scale=args.mask_scale,
                                  threshold_fn=args.threshold_fn,
                                  original=args.no_mask)
    elif args.arch == 'vgg16bn':
        model = net.ModifiedVGG16BN(mask_init=args.mask_init,
                                    mask_scale=args.mask_scale,
                                    threshold_fn=args.threshold_fn,
                                    original=args.no_mask)
    elif args.arch == 'resnet50':   
        model = net.ModifiedResNet(mask_scale=args.mask_scale,
                                   original=args.no_mask)


    else:
        raise ValueError('Architecture %s not supported.' % (args.arch))

     
    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_outputs)
    model.set_dataset(args.dataset)
    if args.cuda:
        model = model.cuda()
    

    # Create the manager object.
    manager = Manager(args, model)

    # Perform necessary mode operations.
    if args.mode == 'finetune':
        if args.no_mask:
            # No masking will be done, used to run baselines of
            # Classifier-Only and Individual Networks.
            # Checks.
            assert args.lr and args.lr_decay_every
            assert not args.lr_mask and not args.lr_mask_decay_every
            assert not args.lr_classifier and not args.lr_classifier_decay_every
            print('No masking, running baselines.')

            # Get optimizer with correct params.
            if args.finetune_layers == 'all':
                params_to_optimize = model.parameters()
            elif args.finetune_layers == 'classifier':
                for param in model.shared.parameters():
                    param.requires_grad = False
                params_to_optimize = model.classifier.parameters()

            # optimizer = optim.Adam(params_to_optimize, lr=args.lr)
            optimizer = optim.SGD(params_to_optimize, lr=args.lr,
                                  momentum=0.9, weight_decay=args.weight_decay)
            optimizers = Optimizers(args)
            optimizers.add(optimizer, args.lr, args.lr_decay_every)
            manager.train(args.finetune_epochs, optimizers,
                          save=True, savename=args.save_prefix)
        else:
            print('Performing KSM masking.')

            # for name, value in model.shared.named_parameters():
            #     print(name)
  
            optimizer_masks = optim.Adam(
                 model.shared.parameters(), lr=args.lr_mask)
            optimizer_classifier = optim.Adam(
                model.classifier.parameters(), lr=args.lr_classifier)       

            optimizers = Optimizers(args)
            optimizers.add(optimizer_masks, args.lr_mask)
            optimizers.add(optimizer_classifier, args.lr_classifier)
              
            manager.train(args.finetune_epochs, optimizers,
                          save=True, savename=args.save_prefix)
    elif args.mode == 'eval':
        # Just run the model on the eval set.
        # load pretrained model
        pretrained = torch.load(args.save_prefix)
        # model.load_state_dict(pretrained['model'])
        manager.eval(0)

        total_mask = 0
        total_mask_one = 0
        for idx, module in enumerate(model.shared.modules()):
            if 'GroupWise' in str(type(module)):
                num_mask = module.mask_real.numel()
                num_one = module.bin_mask.data.eq(1).sum().item()
                total_mask += num_mask
                total_mask_one += num_one
                print('{},{}, -- sparsity:{:2.4f}'.format(idx, module.weight.size(), 1-num_one/num_mask))
        print('total soft factor ratio:{:2.4f}'.format(1-total_mask_one/total_mask))

if __name__ == '__main__':
    main()