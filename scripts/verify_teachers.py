"""Evaluate every vanilla teacher checkpoint in save/models/ on CIFAR-100.

A teacher whose accuracy is below the paper's Table IX teacher row propagates
directly into the student, so verify this before distilling. Expected values
(official RepDistiller checkpoints): wrn_40_2 75.61, resnet56 72.34,
resnet110 74.31, resnet32x4 79.42, vgg13 74.64.

Usage: python scripts/verify_teachers.py [--data_root ./data] [--batch_size 256]
"""
import argparse
import glob
import os
import sys

import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import model_dict  # noqa: E402
from helper.util import AverageMeter, accuracy  # noqa: E402


class _Opt:
    print_freq = 10000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    test_set = tv_datasets.CIFAR100(root=args.data_root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]))
    loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                         shuffle=False, num_workers=4)

    ckpts = sorted(glob.glob(os.path.join('save', 'models', '*_vanilla',
                                          'ckpt_epoch_240.pth')))
    if not ckpts:
        print('no checkpoints found under save/models/*_vanilla/ckpt_epoch_240.pth')
        sys.exit(1)

    for path in ckpts:
        name = os.path.basename(os.path.dirname(path))[:-len('_vanilla')]
        if name not in model_dict:
            print('%-12s unknown model, skipped' % name)
            continue
        model = model_dict[name](num_classes=100)
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            model.load_state_dict(state)
        except Exception as e:
            print('%-12s CORRUPT/incomplete checkpoint (%s)' % (name, e))
            continue
        model.eval()
        top1 = AverageMeter()
        with torch.no_grad():
            for x, y in loader:
                out = model(x)
                acc1, _ = accuracy(out, y, topk=(1, 5))
                top1.update(acc1[0], x.size(0))
        print('%-12s %.2f   (%s)' % (name, top1.avg, path))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
