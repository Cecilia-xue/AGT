import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
        x = input + self.drop_path(x)
        return x


class cx3d(nn.Module):

    def __init__(self, in_chans=1, num_classes=100,
                 depths=[2, 2, 4, 2], dims=[32, 64, 128, 256], drop_path_rate=0., dropout_keep_prob=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0] // 2, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=0, bias=False),
            nn.BatchNorm3d(dims[0] // 2),
            nn.ReLU(),
            nn.Conv3d(dims[0] // 2, dims[0], kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dims[i], eps=1e-6),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.BatchNorm1d(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def cx3d_t(num_classes=100):
    model = cx3d(num_classes=num_classes, depths=[2, 2, 4, 2], dims=[32, 64, 128, 256])
    return model


def cx3d_s(num_classes=100):
    model = cx3d(num_classes=num_classes, depths=[2, 3, 6, 2], dims=[32, 64, 128, 256])
    return model


def cx3d_b(num_classes=100):
    model = cx3d(num_classes=num_classes, depths=[2, 3, 6, 2], dims=[48, 96, 192, 384])
    return model

