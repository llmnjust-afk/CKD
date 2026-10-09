from __future__ import print_function

import torch
import torch.nn as nn


class FeatureFusionModule(nn.Module):
    """CKD feature fusion module F (paper Sec. III-B, Eq. 5).

    Splices the teacher and student penultimate-layer feature
    representations and maps them to fused logits Z^F through a
    trainable projection plus its own classifier. The paper does not
    specify the internal architecture; `arch` selects between the
    single-hidden-layer head (default), a two-hidden-layer MLP, and a
    RepDistiller-VGG-classifier-style head.
    """

    def __init__(self, dim_t, dim_s, num_classes, fused_dim=None,
                 arch='linear', use_bn=True, dropout=0.1):
        super(FeatureFusionModule, self).__init__()
        if fused_dim is None:
            fused_dim = dim_t
        in_dim = dim_t + dim_s

        def block(i, o):
            layers = []
            layers.append(nn.Linear(i, o))
            if use_bn:
                layers.append(nn.BatchNorm1d(o))
            layers.append(nn.ReLU(inplace=True))
            return layers

        if arch == 'linear':
            body = block(in_dim, fused_dim)
        elif arch == 'mlp2':
            body = block(in_dim, fused_dim) + block(fused_dim, fused_dim)
        elif arch == 'vggcls':
            body = block(in_dim, fused_dim) + block(fused_dim, fused_dim) + \
                [nn.Dropout(dropout)]
        else:
            raise ValueError('unknown fusion arch: %s' % arch)
        body = list(body)

        self.proj = nn.Sequential(*body)
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, f_t=None, f_s=None):
        if f_t is None and f_s is None:
            raise ValueError('at least one of f_t / f_s is required')
        if f_t is None:
            z = f_s
        elif f_s is None:
            z = f_t
        else:
            z = torch.cat([f_t, f_s], dim=1)
        feat = self.proj(z)
        logit_f = self.classifier(feat)
        return feat, logit_f
