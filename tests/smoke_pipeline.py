from __future__ import print_function

import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '.')

from models import model_dict
from fuse import FeatureFusionModule
from distiller_zoo import DistillKL, CKDLoss, Exp1Loss, Exp2Loss
from helper.loops_ckd import get_classifier


class _Opt:
    pass


def get_penult_dim(model, input_shape=(2, 3, 32, 32)):
    model.eval()
    with torch.no_grad():
        feats, _ = model(torch.randn(*input_shape), is_feat=True, preact=False)
    return feats[-1].shape[1]


def make_opt():
    o = _Opt()
    o.kd_T = 4.0
    o.alpha = 1.0
    o.beta = 1.0
    o.lam = 1.0
    o.gamma = 1.0
    o.eta = 1.0
    o.alpha1 = 1.0
    o.delta = 1.0
    o.kd_t2 = 1
    o.exp1_literal = 0
    return o


def main():
    torch.manual_seed(0)
    opt = make_opt()

    expected = {
        'vgg13': 512, 'vgg8': 512, 'wrn_40_2': 128, 'wrn_40_1': 64,
        'wrn_16_2': 128, 'resnet56': 64, 'resnet20': 64, 'resnet110': 64,
        'resnet32': 64, 'resnet32x4': 256, 'resnet8x4': 256,
    }
    for name, dim in expected.items():
        got = get_penult_dim(model_dict[name](num_classes=100))
        assert got == dim, 'feature dim mismatch for %s: %d != %d' % (name, got, dim)
    print('[ok] penultimate feature dims for %d models' % len(expected))

    model_t = model_dict['vgg13'](num_classes=100)
    model_s = model_dict['vgg8'](num_classes=100)
    model_t.eval()
    for p in model_t.parameters():
        p.requires_grad_(False)

    dim_t = get_penult_dim(model_t)
    dim_s = get_penult_dim(model_s)
    fusion = FeatureFusionModule(dim_t, dim_s, 100)
    assert fusion.proj[0].in_features == dim_t + dim_s
    assert fusion.classifier.out_features == 100

    x = torch.randn(16, 3, 32, 32)
    target = torch.randint(0, 100, (16,))

    with torch.no_grad():
        feat_t_list, logit_t = model_t(x, is_feat=True, preact=False)
    feat_t = feat_t_list[-1]

    ckdl = CKDLoss(opt)
    kdl = DistillKL(opt.kd_T)
    e1 = Exp1Loss(opt)
    opt.exp1_literal = 1
    e1_lit = Exp1Loss(opt)
    e2 = Exp2Loss(opt)
    ce = nn.CrossEntropyLoss()

    trainable = list(model_s.parameters()) + list(fusion.parameters())
    names = [('s', p) for p in model_s.parameters()] + [('f', p) for p in fusion.parameters()]

    first_losses = {}
    for method in ['ckd', 'kd', 'exp1', 'exp1_literal', 'exp2']:
        ms = copy.deepcopy(model_s)
        fs = copy.deepcopy(fusion)
        ms.train(), fs.train()
        optimizer = optim.SGD([{'params': ms.parameters()}, {'params': fs.parameters()}], lr=0.1, momentum=0.9)
        losses = []
        regs = []
        for step in range(4):
            with torch.no_grad():
                ft, lt = model_t(x, is_feat=True, preact=False)
            ft = ft[-1]
            fsel, ls = ms(x, is_feat=True, preact=False)
            fsel = fsel[-1]
            _, lf = fs(ft, fsel)
            if method == 'ckd':
                loss, info = ckdl(lt, ls, lf, target)
                assert abs(info['loss_f'] + info['loss_s'] - loss.item()) < 1e-4
            elif method == 'kd':
                loss = kdl(ls, lt)
            elif method == 'exp1':
                loss, info = e1(ft, fsel, ls, target)
                regs.append(info['reg'])
            elif method == 'exp1_literal':
                loss, info = e1_lit(ft, fsel, ls, target)
                regs.append(info['reg'])
            else:
                fa = ft + fsel
                la = get_classifier(model_t)(fa)
                loss, info = e2(la, ls, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]
        assert losses[-1] < losses[0], 'loss did not decrease for %s: %s' % (method, losses)
        if method == 'exp1_literal':
            spread = max(regs) - min(regs)
            assert spread < 1e-5, 'literal Eq.10 reg must be constant (||F^T||): %s' % regs
            print('     (literal Eq.10 reg constant at %.4f -> alpha1 has no effect; hint form used by default)' % regs[0])
        n_grad = sum(1 for g in [p.grad for p in ms.parameters()] if g is not None)
        assert n_grad > 0
        print('[ok] method %-12s loss %s' % (method, ['%.3f' % l for l in losses]))

    fa = feat_t + feat_t
    assert get_classifier(model_t)(fa).shape == (16, 100)
    print('[ok] Exp2 teacher-classifier path')

    print('ALL PIPELINE SMOKE TESTS PASSED')


if __name__ == '__main__':
    main()
