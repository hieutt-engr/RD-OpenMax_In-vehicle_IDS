import torch
import torch.nn as nn
import timm

from attentions.SEAttention import SEAttention
from attentions.CBAM import CBAMBlock
from representation.MPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer

__all__ = ['efficientnet']

class AttentionBlock(nn.Module):
    def __init__(self, inplanes, planes, att_dim=64):
        super(AttentionBlock, self).__init__()
        self.ch_dim = att_dim

        self.conv_for_DR = nn.Conv2d(inplanes, self.ch_dim, kernel_size=1, stride=1, bias=False)
        self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(int(self.ch_dim * (self.ch_dim + 1) / 2), planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.conv_for_DR(x)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = CovpoolLayer(out)
        out = SqrtmLayer(out, 5)
        out = TriuvecLayer(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1, 1).contiguous()
        out = self.sigmoid(out)

        return residual * out


class EfficientNetWithRepresentation(nn.Module):
    def __init__(self,
                 model_name='efficientnetv2_m',
                 num_classes=6,
                 attention='Cov',
                 pretrained=False,
                 input_size=64):
        super(EfficientNetWithRepresentation, self).__init__()

        # 1. Backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.out_channels = self.backbone.feature_info[-1]['num_chs']  # usually 1280 or 1408
        self.input_dim = self.out_channels  # used for representation methods

        # 2. Attention Module (optional)
        self.attention = nn.Identity()
        if attention == 'Cov':
            self.attention = AttentionBlock(inplanes=self.out_channels, planes=self.out_channels)
        elif attention == 'SEAttention':
            self.attention = SEAttention(channel=self.out_channels, reduction=16)
        elif attention == 'CBAM':
            self.attention = CBAMBlock(channel=self.out_channels, reduction=8, kernel_size=3)
        elif attention is not None:
            raise NotImplementedError(f"Unknown attention type: {attention}")

        # 3. Representation layer placeholder (to be injected by Newmodel)
        self.representation = nn.Identity()
        self.representation_dim = self.out_channels

        # 4. Classifier
        self.fc = nn.Linear(self.representation_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)[-1]  # [B, C, H, W]

        x = self.attention(x)

        x = self.representation(x)  # [B, D] or [B, C, H, W]
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        return self.fc(x)


def efficientnet(pretrained=False, **kwargs):
    model = EfficientNetWithRepresentation(pretrained=pretrained, **kwargs)
    return model