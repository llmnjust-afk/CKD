from __future__ import print_function

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms

from models import model_dict
from dataset.cifar100 import CIFAR100Instance
from dataset.cifar10 import CIFAR10Instance
from dataset.imagenet import ImageFolderInstance, get_test_loader
from torchvision import datasets as tv_datasets
from fuse import FeatureFusionModule
from helper.loops import validate
from helper.loops_ckd import train_ckd
from helper.util import adjust_learning_rate, AverageMeter, accuracy
from distiller_zoo import DistillKL, CKDLoss, Exp1Loss, Exp2Loss


def parse_option():
    parser = argparse.ArgumentParser('argument for CKD distillation')

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=None)

    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--lr_decay_epochs', type=str, default=None)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10', 'imagenet'])
    parser.add_argument('--path_t', type=str, default=None)
    parser.add_argument('--model_t', type=str, default=None)
    parser.add_argument('--model_s', type=str, default='resnet8')

    parser.add_argument('--method', type=str, default='ckd',
                        choices=['ckd', 'kd', 'exp1', 'exp2'])

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--alpha1', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--kd_T', type=float, default=4.0)
    parser.add_argument('--kd_t2', type=int, default=1)
    parser.add_argument('--exp1_literal', type=int, default=0)
    parser.add_argument('--full_grad_kl', type=int, default=0,
                        help='1: literal autograd of Eqs.6-8 (KLs propagate to every probability incl. the student side); 0: F.kl_div convention (target side detached)')
    parser.add_argument('--fusion_dim', type=str, default='teacher',
                        help="'teacher' | 'student' | explicit integer width")
    parser.add_argument('--fusion_arch', type=str, default='linear',
                        choices=['linear', 'mlp2', 'vggcls'])
    parser.add_argument('--val_fusion', type=int, default=1,
                        help='log per-epoch test accuracy of the fusion module (diagnostic)')

    parser.add_argument('--subset', type=float, default=1.0)
    parser.add_argument('--trial', type=str, default='1')
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--save_dir', type=str, default='./save/student_data')
    parser.add_argument('--seed', type=int, default=0)

    opt = parser.parse_args()

    if opt.epochs is None:
        opt.epochs = 240 if opt.dataset != 'imagenet' else 90
    if opt.learning_rate is None:
        opt.learning_rate = 0.1 if opt.dataset != 'imagenet' else 0.2
    if opt.lr_decay_epochs is None:
        opt.lr_decay_epochs = '150,180,210' if opt.dataset != 'imagenet' else '30,60,90'
    if opt.weight_decay is None:
        opt.weight_decay = 5e-4 if opt.dataset != 'imagenet' else 1e-4

    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]

    opt.input_shape = (2, 3, 32, 32) if opt.dataset in ['cifar100', 'cifar10'] else (2, 3, 224, 224)

    if opt.path_t:
        t_label = os.path.basename(os.path.dirname(opt.path_t))
    elif opt.model_t:
        t_label = opt.model_t
    else:
        t_label = 'random_teacher'
    opt.model_name = '%s_%s_%s_T%s_trial_%s' % (
        opt.dataset, opt.model_s, opt.method, t_label, opt.trial)

    opt.save_folder = os.path.join(opt.save_dir, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def get_teacher_name(model_path):
    name = os.path.basename(os.path.dirname(model_path))
    for suffix in ['_vanilla', '_expt']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def load_teacher(model_path, n_cls, model_name=None):
    print('=> loading teacher model from:', model_path)
    name = model_name or get_teacher_name(model_path)
    if name not in model_dict:
        raise ValueError('unknown teacher model: %s' % name)
    model_t = model_dict[name](num_classes=n_cls)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model_t.load_state_dict(state)
    print('=> done')
    return model_t


def validate_fusion(val_loader, model_t, model_s, fusion, opt):
    fusion.eval()
    model_s.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for data in val_loader:
            input, target = data[0], data[1]
            input = input.float().to(opt.device)
            target = target.to(opt.device)
            feat_t, _ = model_t(input, is_feat=True, preact=False)
            feat_s, _ = model_s(input, is_feat=True, preact=False)
            _, logit_f = fusion(feat_t[-1], feat_s[-1])
            acc1 = accuracy(logit_f, target, topk=(1,))[0]
            top1.update(acc1[0], input.size(0))
    return top1.avg


def get_penult_dim(model, input_shape):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        feats, _ = model(torch.randn(*input_shape).to(device), is_feat=True, preact=False)
    return feats[-1].shape[1]


def build_dataloaders(opt):
    if opt.dataset in ['cifar100', 'cifar10']:
        if opt.dataset == 'cifar100':
            base_set = CIFAR100Instance
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        else:
            base_set = CIFAR10Instance
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = base_set(root=opt.data_root, download=True, train=True,
                             transform=train_transform)
        test_cls = tv_datasets.CIFAR100 if opt.dataset == 'cifar100' else tv_datasets.CIFAR10
        test_set = test_cls(root=opt.data_root, download=True, train=False,
                            transform=test_transform)
        if opt.subset < 1.0:
            rng = np.random.RandomState(42)
            n = len(train_set)
            idx = rng.choice(n, int(n * opt.subset), replace=False)
            train_set = Subset(train_set, idx.tolist())
        train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
        test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                                 shuffle=False, num_workers=opt.num_workers)
        n_cls = 100 if opt.dataset == 'cifar100' else 10
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        data_folder = os.path.join(opt.data_root, 'imagenet')
        train_set = ImageFolderInstance(root=os.path.join(data_folder, 'train'),
                                        transform=train_transform)
        if opt.subset < 1.0:
            rng = np.random.RandomState(42)
            n = len(train_set)
            idx = rng.choice(n, int(n * opt.subset), replace=False)
            train_set = Subset(train_set, idx.tolist())
        train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers,
                                  pin_memory=True)
        test_loader = get_test_loader(dataset='imagenet',
                                      batch_size=opt.batch_size,
                                      num_workers=opt.num_workers)
        n_cls = 1000
    return train_loader, test_loader, n_cls


def main():
    best_acc = 0.0
    opt = parse_option()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
        cudnn.benchmark = True

    train_loader, val_loader, n_cls = build_dataloaders(opt)

    if opt.dataset == 'imagenet' and not opt.path_t:
        name = opt.model_t or 'imagenet_resnet34'
        model_t = model_dict[name](num_classes=n_cls, pretrained=True)
    else:
        model_t = load_teacher(opt.path_t, n_cls, model_name=opt.model_t)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    if torch.cuda.is_available():
        opt.device = 'cuda'
    elif torch.backends.mps.is_available():
        opt.device = 'mps'
    else:
        opt.device = 'cpu'
    model_t = model_t.to(opt.device)
    model_s = model_s.to(opt.device)
    model_t.eval()
    for p in model_t.parameters():
        p.requires_grad_(False)

    dim_t = get_penult_dim(model_t, opt.input_shape)
    dim_s = get_penult_dim(model_s, opt.input_shape)
    fused_dim = dim_t
    if opt.fusion_dim == 'student':
        fused_dim = dim_s
    elif opt.fusion_dim not in ['teacher', 'student']:
        fused_dim = int(opt.fusion_dim)
    fusion = FeatureFusionModule(dim_t, dim_s, n_cls, fused_dim=fused_dim,
                                 arch=opt.fusion_arch).to(opt.device)
    print('teacher penult dim: %d, student penult dim: %d, fused dim: %d'
          % (dim_t, dim_s, fused_dim))

    if opt.method == 'ckd':
        criterion = CKDLoss(opt)
    elif opt.method == 'kd':
        criterion = DistillKL(opt.kd_T)
    elif opt.method == 'exp1':
        criterion = Exp1Loss(opt)
    else:
        criterion = Exp2Loss(opt)
    criterion = criterion.to(opt.device)

    module_list = nn.ModuleList([model_t, model_s, fusion])
    trainable_list = nn.ModuleList([model_s, fusion])

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion_cls = nn.CrossEntropyLoss().to(opt.device)
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: %.3f' % teacher_acc)

    with open(os.path.join(opt.save_folder, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=2)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> epoch %d, learning rate %f" % (epoch, optimizer.param_groups[0]['lr']))

        time1 = time.time()
        loss, info = train_ckd(epoch, train_loader, module_list, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch %d, total time %.2f, loss %.4f' % (epoch, time2 - time1, loss))
        for k in sorted(info.keys()):
            print('   %s: %.4f' % (k, info[k].avg))

        test_acc, test_acc_top5, _ = validate(val_loader, model_s, criterion_cls, opt)

        fusion_acc = -1.0
        if opt.val_fusion and opt.method == 'ckd':
            fusion_acc = validate_fusion(val_loader, model_t, model_s, fusion, opt)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'opt': vars(opt),
                'epoch': epoch,
                'model': model_s.state_dict(),
                'fusion': fusion.state_dict(),
                'best_acc': test_acc,
            }
            torch.save(state, os.path.join(opt.save_folder, '%s_best.pth' % opt.model_s))

        print('epoch %d, student test acc %.3f (top5 %.3f), fusion test acc %.3f, best %.3f'
              % (epoch, test_acc, test_acc_top5, fusion_acc, best_acc))
        sys.stdout.flush()

    print('best accuracy over %d epochs: %.3f' % (opt.epochs, best_acc))
    with open(os.path.join(opt.save_folder, 'result.txt'), 'w') as f:
        f.write('best_acc %.3f\n' % best_acc)


if __name__ == '__main__':
    main()
