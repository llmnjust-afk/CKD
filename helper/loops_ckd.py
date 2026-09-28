from __future__ import print_function, division

import time
import sys
import torch

from .util import AverageMeter, accuracy


def get_classifier(m):
    """Return the final classification layer of a model zoo network."""
    for name in ['fc', 'linear', 'classifier']:
        mod = getattr(m, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    raise ValueError('cannot locate classifier of model %s' % type(m).__name__)


def train_ckd(epoch, train_loader, module_list, criterion, optimizer, opt):
    """One epoch of CKD training.

    module_list = [model_t (frozen), model_s (trainable), fusion (trainable)]
    """
    model_t = module_list[0]
    model_s = module_list[1]
    fusion = module_list[2]

    logit_f = None
    model_s.train()
    fusion.train()
    model_t.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    info_meters = {}
    end = time.time()

    for idx, data in enumerate(train_loader):
        input, target = data[0], data[1]
        data_time.update(time.time() - end)

        input = input.float().to(opt.device)
        target = target.to(opt.device)

        with torch.no_grad():
            feat_t_list, logit_t = model_t(input, is_feat=True, preact=False)
        feat_t = feat_t_list[-1].detach()

        feat_s_list, logit_s = model_s(input, is_feat=True, preact=False)
        feat_s = feat_s_list[-1]

        if opt.method == 'ckd':
            _, logit_f = fusion(feat_t, feat_s)
            loss, info = criterion(logit_t, logit_s, logit_f, target)
        elif opt.method == 'kd':
            loss, info = criterion(logit_s, logit_t)
        elif opt.method == 'exp1':
            loss, info = criterion(feat_t, feat_s, logit_s, target)
        elif opt.method == 'exp2':
            if feat_t.shape != feat_s.shape:
                raise ValueError('Exp2 requires equal teacher/student feature dims')
            feat_a = feat_t + feat_s
            logit_a = get_classifier(model_t)(feat_a)
            loss, info = criterion(logit_a, logit_s, target)
        else:
            raise NotImplementedError(opt.method)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        for k, v in info.items():
            if k not in info_meters:
                info_meters[k] = AverageMeter()
            info_meters[k].update(v, input.size(0))

        acc_s = accuracy(logit_s, target)[0]
        info_meters.setdefault('train_acc_s', AverageMeter()).update(acc_s[0], input.size(0))
        if opt.method == 'ckd':
            acc_f = accuracy(logit_f, target)[0]
            info_meters.setdefault('train_acc_f', AverageMeter()).update(acc_f[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
            extra = '  '.join('{0} {1.val:.4f} ({1.avg:.4f})'.format(k, m)
                              for k, m in sorted(info_meters.items()))
            print('   ' + extra)
            sys.stdout.flush()

    return losses.avg, info_meters
