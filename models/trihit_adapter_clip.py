from models.adapter_modules import *
from models.trihit_x import trihit_x


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class conv_block(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x, type='channel_first'):
        if type == 'channel_first':
            input = x
            x = self.dwconv(x)
            x = self.norm(x)
            x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
            x = input + x
        elif type == 'channel_last':
            input = x
            x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
            x = self.dwconv(x)
            x = self.norm(x)
            x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = input + x
        return x


class conv_block_ns(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x, type='channel_first'):
        if type == 'channel_first':
            x = self.dwconv(x)
            x = self.norm(x)
            x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
        elif type == 'channel_last':
            x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
            x = self.dwconv(x)
            x = self.norm(x)
            x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
        return x


# zero convolution adapter: sequence arch
class trihit_clip_adapter(trihit_x):
    def __init__(self,
                 in_chans=1,
                 num_classes=100,
                 depths=[2, 2, 4, 2],
                 dims=[64, 128, 320, 512],
                 dp_rate=0.15,
                 arch=['conv', 'hit', 'hit', 'hit'],
                 head_dropout=0.0,
                 ):

        super().__init__(in_chans=in_chans, depths=depths, dims=dims, drop_path_rate=dp_rate, arch=arch, add_classifer=False)

        self.adapter_layers = nn.ModuleList()
        for dim in dims:
            adapter_layer = zero_module(conv_block(dim=dim))
            self.adapter_layers.append(adapter_layer)

        self.adapter_cls_norm = nn.LayerNorm(dims[-1])
        self.adapter_cls_head = MLP_head(dims[-1], num_classes, head_dropout=head_dropout)

    def forward(self, x):
        convert_flag = False
        for i in range(self.layer_num):
            if self.arch[i] == 'conv':
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                x = self.adapter_layers[i](x, 'channel_first')
            elif 'hit' in self.arch[i]:
                if convert_flag: x = rearrange(x, 'b s h w c-> b c s h w')
                x = self.downsample_layers[i](x)
                x = rearrange(x, 'b c s h w -> b s h w c')
                x = self.stages[i](x)
                x = self.adapter_layers[i](x, 'channel_last')
                convert_flag = True

        x = x.mean([1, 2, 3])
        x = self.adapter_cls_norm(x)
        out = self.adapter_cls_head(x)

        return out

def trihit_cth_clip(num_classes=100, dp_rate=0.15):
    model = trihit_clip_adapter(num_classes=num_classes, depths=[2, 2, 4, 2], dims=[32, 64, 128, 256], dp_rate=dp_rate, arch=['conv', 'hit', 'hit', 'hit'],)
    return model

