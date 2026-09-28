from __future__ import print_function

import torch
import torch.nn as nn
import torchvision


class ImageNetResNet(nn.Module):
    """torchvision ResNet wrapper exposing penultimate features and a
    separately callable classifier (needed by Exp2 / the fusion module).

    torchvision IMAGENET1K_V1 accuracies match the paper's Table X:
    resnet34 -> 73.31 top-1 (paper teacher), resnet18 -> 69.75 top-1
    (paper student baseline).
    """

    def __init__(self, arch='resnet34', pretrained=True, num_classes=1000):
        super(ImageNetResNet, self).__init__()
        if pretrained:
            m = torchvision.models.__dict__[arch](
                num_classes=num_classes, weights='IMAGENET1K_V1')
        else:
            m = torchvision.models.__dict__[arch](
                num_classes=num_classes, weights=None)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
        self.classifier = m.fc
        self.feat_dim = self.classifier.weight.shape[1]

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        out = self.classifier(feat)
        if is_feat:
            return [feat], out
        return out


def imagenet_resnet34(pretrained=True, num_classes=1000):
    return ImageNetResNet('resnet34', pretrained, num_classes)


def imagenet_resnet18(pretrained=False, num_classes=1000):
    return ImageNetResNet('resnet18', pretrained, num_classes)
