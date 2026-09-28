from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class CKDLoss(nn.Module):
    """Collaborative knowledge distillation losses (paper Eqs. 6-8).

    Eq. (6): Loss_F = alpha * KL(P^T || P^F) + beta * KL(P^S || P^F)
                      + lambda * CE(Y, P^F)
    Eq. (7): Loss_S = gamma * KL(P^F || P^S) + eta * CE(Y, P^S)
    Eq. (8): Loss = Loss_F + Loss_S

    Probabilities are computed with temperature tau (opt.kd_T); the paper
    states tau = 1 for cross-entropy, so CE terms always use tau = 1.
    When opt.kd_t2 is set, KL terms are scaled by tau^2 (tau^2 is omitted
    from the printed equations but is the standard convention on the
    RepDistiller/DKD benchmark that produced the paper's Table IX-X).
    All KL directions follow the printed equations exactly.
    """

    def __init__(self, opt):
        super(CKDLoss, self).__init__()
        self.T = opt.kd_T
        self.alpha = opt.alpha
        self.beta = opt.beta
        self.lam = opt.lam
        self.gamma = opt.gamma
        self.eta = opt.eta
        self.t2 = opt.kd_t2

    def forward(self, logit_t, logit_s, logit_f, target):
        T = self.T
        p_t = F.softmax(logit_t / T, dim=1)
        log_p_s = F.log_softmax(logit_s / T, dim=1)
        p_s = F.softmax(logit_s / T, dim=1)
        log_p_f = F.log_softmax(logit_f / T, dim=1)
        p_f = F.softmax(logit_f / T, dim=1)

        scale = T * T if self.t2 else 1.0

        kl_tf = F.kl_div(log_p_f, p_t, reduction='batchmean')
        kl_sf = F.kl_div(log_p_f, p_s, reduction='batchmean')
        kl_fs = F.kl_div(log_p_s, p_f, reduction='batchmean')

        ce_f = F.cross_entropy(logit_f, target)
        ce_s = F.cross_entropy(logit_s, target)

        loss_f = self.alpha * scale * kl_tf + self.beta * scale * kl_sf + self.lam * ce_f
        loss_s = self.gamma * scale * kl_fs + self.eta * ce_s
        loss = loss_f + loss_s

        info = {
            'loss_f': loss_f.item(),
            'loss_s': loss_s.item(),
            'kl_tf': kl_tf.item(),
            'kl_sf': kl_sf.item(),
            'kl_fs': kl_fs.item(),
            'ce_f': ce_f.item(),
            'ce_s': ce_s.item(),
        }
        return loss, info


class Exp1Loss(nn.Module):
    """Exp1 loss (paper Appendix A, Eqs. 9-10).

    Eq. (9):  F^A = F^T + F^S
    Eq. (10): Loss_Exp1 = alpha_1/2 * ||F^S - F^A||_2 + beta * CE(Y, P^S)

    With the printed F^A the first term equals alpha_1/2 * ||F^T||_2, a
    constant w.r.t. trainable parameters. We therefore default to the
    (FitNet-based, as stated in Appendix A) matching form
    alpha_1/2 * ||F^S - F^T||_2 and keep the literal printed form behind
    `literal=True` for exact reproduction of the printed equation.
    """

    def __init__(self, opt):
        super(Exp1Loss, self).__init__()
        self.alpha1 = opt.alpha1
        self.beta = opt.beta
        self.literal = getattr(opt, 'exp1_literal', False)

    def forward(self, feat_t, feat_s, logit_s, target):
        if feat_t.shape != feat_s.shape:
            raise ValueError('Exp1 requires equal teacher/student feature dims')
        if self.literal:
            f_a = feat_t + feat_s
            reg = (feat_s - f_a).norm(2)
        else:
            reg = (feat_s - feat_t).norm(2)
        ce = F.cross_entropy(logit_s, target)
        loss = self.alpha1 / 2.0 * reg + self.beta * ce
        info = {'reg': reg.item(), 'ce_s': ce.item()}
        return loss, info


class Exp2Loss(nn.Module):
    """Exp2 loss (paper Appendix A, Eqs. 11-12).

    Eq. (11): Z^A = sigma(F^A), sigma = teacher classifier
    Eq. (12): Loss_Exp2 = gamma * KL(P^A || P^S) + delta * CE(Y, P^S)
    """

    def __init__(self, opt):
        super(Exp2Loss, self).__init__()
        self.gamma = opt.gamma
        self.delta = opt.delta
        self.T = opt.kd_T
        self.t2 = opt.kd_t2

    def forward(self, logit_a, logit_s, target):
        T = self.T
        p_a = F.softmax(logit_a / T, dim=1)
        log_p_s = F.log_softmax(logit_s / T, dim=1)
        scale = T * T if self.t2 else 1.0
        kl = F.kl_div(log_p_s, p_a, reduction='batchmean')
        ce = F.cross_entropy(logit_s, target)
        loss = self.gamma * scale * kl + self.delta * ce
        info = {'kl': kl.item(), 'ce_s': ce.item()}
        return loss, info
