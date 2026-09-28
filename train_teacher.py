from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from helper.util import adjust_learning_rate
from helper.loops import train_vanilla as train, validate


def parse_option():
    parser = argparse.ArgumentParser('argument for training vanilla teachers')

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=240)

    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44',
                                 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10'])
    parser.add_argument('-t', '--trial', type=str, default='0')
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--save_dir', type=str, default='./save/models')
    parser.add_argument('--subset', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)

    opt = parser.parse_args()

    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]
    opt.model_name = '%s_%s_vanilla_trial_%s' % (opt.model, opt.dataset, opt.trial)
    opt.save_folder = os.path.join(opt.save_dir, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def main():
    best_acc = 0
    opt = parse_option()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
        cudnn.benchmark = True

    if opt.dataset == 'cifar100':
        train_loader, val_loader, _ = get_cifar100_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader, _ = get_cifar10_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    if opt.subset < 1.0:
        rng = np.random.RandomState(42)
        ds = train_loader.dataset
        n = len(ds)
        idx = rng.choice(n, int(n * opt.subset), replace=False)
        train_loader = DataLoader(Subset(ds, idx.tolist()),
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    model = model_dict[opt.model](num_classes=n_cls)
    model = model.cuda() if torch.cuda.is_available() else model

    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    with open(os.path.join(opt.save_folder, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=2)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print('==> epoch %d, learning rate %f' % (epoch, optimizer.param_groups[0]['lr']))

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch %d, total time %.2f, train acc %.3f' % (epoch, time2 - time1, train_acc))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(state, os.path.join(opt.save_folder, '%s_best.pth' % opt.model))

        if epoch % opt.save_freq == 0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
            }
            torch.save(state, os.path.join(opt.save_folder,
                                           'ckpt_epoch_%d.pth' % epoch))

        print('epoch %d, test acc %.3f (top5 %.3f), best test acc %.3f'
              % (epoch, test_acc, test_acc_top5, best_acc))
        sys.stdout.flush()

    print('best accuracy over %d epochs: %.3f' % (opt.epochs, best_acc))
    with open(os.path.join(opt.save_folder, 'result.txt'), 'w') as f:
        f.write('best_acc %.3f\n' % best_acc)


if __name__ == '__main__':
    main()
