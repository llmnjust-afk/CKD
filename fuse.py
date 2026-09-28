from __future__ import print_function

import torch
import torch.nn as nn


class FeatureFusionModule(nn.Module):
    """CKD feature fusion module F (paper Sec. III-B, Eq. 5).

    Splices the teacher and student penultimate-layer feature
    representations and maps them to fused logits Z^F through a
    trainable projection plus its own classifier.
    """

    def __init__(self, dim_t, dim_s, num_classes, fused_dim=None, use_bn=True):
        super(FeatureFusionModule, self).__init__()
        if fused_dim is None:
            fused_dim = dim_t
        self.proj = nn.Sequential(
            nn.Linear(dim_t + dim_s, fused_dim),
            nn.BatchNorm1d(fused_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
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
