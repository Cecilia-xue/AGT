from models.adapter_modules import *
from models.trihit_x import trihit_x

def centercrop(x, crop_size):
    b, c, l, h, w = x.shape
    bound = (h - crop_size) // 2
    hs, he = bound, bound + crop_size
    ws, we = bound, bound + crop_size
    crop_x = x[:,:,:, hs:he, ws:we]
    return crop_x

class MLP_3d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim, bias=False)  # pointwise/1x1 convs, implemented with linear layers
        self.act = StarReLU()
        self.pwconv2 = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        input = x # N L H W C
        x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x

class hitblock(nn.Module):
    """
    Triple structured hyperspectral image transformer block (trihit block),
    spectral attention and spatial attention are conducted in parallel way.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()

        # spectral attention
        self.norm1 = norm_layer(dim)
        self.spectral_attn = Attention(dim=dim)

        # spatial attention
        self.norm2 = norm_layer(dim)
        self.spatial_attn = Attention(dim=dim)

        # IRFFN
        self.norm3 = norm_layer(dim)
        self.mlp = MLP_3d(dim=dim)

    def forward(self, x):
        x_spe = x + self.spectral_attn(self.norm1(x), 'spectral')
        x_spa = x + self.spatial_attn(self.norm2(x), 'spatial')
        x = x_spe + x_spa
        x = x + self.mlp(self.norm3(x))
        return x


class hitlayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        # blocks
        self.blocks = nn.Sequential(*[
            hitblock(dim=dim)
            for j in range(depth)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x

"""
num_classes=num_classes,
bk_depths=[2, 2, 4, 2], bk_dims=[32, 64, 128, 256], bk_arch=['conv', 'hit', 'hit', 'hit'],
dp_rate=dp_rate,
ladder_depths=[1,1,2,1], ladder_dims=[32, 64, 128, 256], ladder_input_size=13
"""


class trihit_lst(trihit_x):
    def __init__(self,
                 in_chans=1,
                 num_classes=100,
                 bk_depths=[2, 2, 4, 2],
                 bk_dims=[32, 64, 128, 256],
                 bk_arch=['conv', 'hit', 'hit', 'hit'],
                 dp_rate=0.15,
                 ladder_depths=[1, 2, 1],
                 ladder_dims=[64, 128, 256],
                 ladder_input_size=13,
                 head_dropout=0.0,
                 ):

        super().__init__(in_chans=in_chans, depths=bk_depths, dims=bk_dims, drop_path_rate=dp_rate, arch=bk_arch, add_classifer=False)

        self.adapter_downsample_layers = nn.ModuleList()
        self.adapter_stages = nn.ModuleList()
        self.adapter_bridge = nn.ModuleList()
        self.ladder_input_size = ladder_input_size

        stem = nn.Sequential(
            nn.Conv3d(in_chans, ladder_dims[0] // 2, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(ladder_dims[0] // 2),
            StarReLU(),
            nn.Conv3d(ladder_dims[0] // 2, ladder_dims[0], kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(ladder_dims[0]),
        )
        self.adapter_downsample_layers.append(stem)

        for i in range(2):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(ladder_dims[i]),
                nn.Conv3d(ladder_dims[i], ladder_dims[i + 1], kernel_size=3, stride=2, padding=0),
            )
            self.adapter_downsample_layers.append(downsample_layer)

        for i in range(3):
            stage = hitlayer(dim=ladder_dims[i], depth=ladder_depths[i])
            self.adapter_stages.append(stage)

        for i in range(3):
            bridge = nn.Sequential(
                nn.Linear(ladder_dims[i], ladder_dims[i], bias=False),
                nn.LayerNorm(ladder_dims[i])
            )
            self.adapter_bridge.append(bridge)

        self.adapter_cls_norm = nn.LayerNorm(ladder_dims[-1])
        self.adapter_cls_head = MLP_head(ladder_dims[-1], num_classes, head_dropout=head_dropout)

    def forward(self, x):
        bk_x = x
        ladder_x = centercrop(x, self.ladder_input_size)
        bk_x = self.downsample_layers[0](bk_x)
        bk_x = self.stages[0](bk_x)

        for i, [adapter_down, adapter_stage, bk_down, bk_stage, bridge] in \
                enumerate(zip(self.adapter_downsample_layers, self.adapter_stages,
                          self.downsample_layers[1:], self.stages[1:], self.adapter_bridge)):
            if i>0:
                bk_x = rearrange(bk_x, 'b s h w c-> b c s h w')
                ladder_x = rearrange(ladder_x, 'b s h w c-> b c s h w')

            bk_x = bk_down(bk_x)
            bk_x = rearrange(bk_x, 'b c s h w -> b s h w c')
            bk_x = bk_stage(bk_x)

            ladder_x = adapter_down(ladder_x)
            ladder_x = rearrange(ladder_x, 'b c s h w -> b s h w c')

            ladder_x = ladder_x + bk_x
            ladder_x = adapter_stage(ladder_x)

        x = ladder_x.mean([1,2,3])
        x = self.adapter_cls_norm(x)
        out = self.adapter_cls_head(x)

        return out

"""
following the architecure proposed in LST, we fuse features from different branches by adding them element by element
"""

def trihit_cth_lst(num_classes=100, dp_rate=0.15):
    model = trihit_lst(num_classes=num_classes,
                           bk_depths=[2, 2, 4, 2], bk_dims=[32, 64, 128, 256], bk_arch=['conv', 'hit', 'hit', 'hit'],
                           dp_rate=dp_rate,
                           ladder_depths=[1,2,1], ladder_dims=[64, 128, 256], ladder_input_size=13)
    return model

