from models.basic_operators import *
from timm.models.layers import DropPath, trunc_normal_

class MLP_3d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=False)  # pointwise/1x1 convs, implemented with linear layers
        self.act = StarReLU()
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=False)

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

    def __init__(self, dim, norm_layer=nn.LayerNorm, drop_path=0.):
        super().__init__()

        # spectral attention
        self.norm1 = norm_layer(dim)
        self.spectral_attn = Attention(dim=dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # spatial attention
        self.norm2 = norm_layer(dim)
        self.spatial_attn = Attention(dim=dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # IRFFN
        self.norm3 = norm_layer(dim)
        self.mlp = MLP_3d(dim=dim)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_spe = x + self.drop_path1(self.spectral_attn(self.norm1(x), 'spectral'))
        x_spa = x + self.drop_path2(self.spatial_attn(self.norm2(x), 'spatial'))
        x = x_spe + x_spa
        x = x + self.drop_path3(self.mlp(self.norm3(x)))
        return x


class hitlayer(nn.Module):
    def __init__(self, dim, depth, dp_rates):
        super().__init__()
        # blocks
        self.blocks = nn.Sequential(*[
            hitblock(dim=dim, drop_path=dp_rates[j])
            for j in range(depth)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x

#******************************************************************************************************convblock convlayer
class convblock(nn.Module):

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = StarReLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
        x = input + self.drop_path(x)
        return x

class convlayer(nn.Module):
    def __init__(self, dim, depth, dp_rates):
        super().__init__()
        # blocks
        self.blocks = nn.Sequential(*[
            convblock(dim=dim, drop_path=dp_rates[j])
            for j in range(depth)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x



class trihit_x(nn.Module):
    """
    Triple structured hyperspectral image transformer (trihit).
    """
    def __init__(self, in_chans=1, num_classes=100,
                 depths=[2, 2, 4, 2],
                 dims=[32, 64, 128, 256],
                 drop_path_rate=0.15,
                 arch = ['conv', 'hit', 'hit', 'hit'],
                 head_dropout=0.0,
                 add_classifer=True,
                 **kwargs,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and three downsample layers
        self.stages=nn.ModuleList() # trihit layers
        self.layer_num = len(depths)
        self.arch = arch

        # downsample layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0] // 2, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(dims[0] // 2),
            StarReLU(),
            nn.Conv3d(dims[0] // 2, dims[0], kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 0, 0), bias=False),
            nn.BatchNorm3d(dims[0])
        )

        self.downsample_layers.append(stem)

        for i in range(self.layer_num-1):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=0),
            )
            self.downsample_layers.append(downsample_layer)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layer_dpr = []
        cur_depth = 0
        for depth in depths:
            layer_dpr.append(dpr[cur_depth:cur_depth + depth])
            cur_depth += depth

        for i in range(self.layer_num):
            if self.arch[i] == 'conv':
                layer = convlayer
            elif self.arch[i] == 'hit':
                layer = hitlayer
            else:
                print('undifined type {}'.format(self.arch[i]))

            stage=layer(dim=dims[i], depth=depths[i], dp_rates=layer_dpr[i])
            self.stages.append(stage)

        # Classifier head
        self.add_classifier = add_classifer
        if add_classifer:
            self.cls_norm = nn.LayerNorm(dims[-1])
            self.cls_head = MLP_head(dims[-1], num_classes, head_dropout=head_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        convert_flag=False
        for i in range(self.layer_num):
            if self.arch[i] == 'conv':
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            elif 'hit' in self.arch[i]:
                if convert_flag: x = rearrange(x, 'b s h w c-> b c s h w')
                x = self.downsample_layers[i](x)
                x = rearrange(x, 'b c s h w -> b s h w c')
                x = self.stages[i](x)
                convert_flag=True

        return x.mean([1, 2, 3]) # global average pooling, (b s h w c) -> (b c)

    def forward(self, x):
        x = self.forward_features(x)
        if self.add_classifier:
            x = self.cls_head(self.cls_norm(x))
        return x


# trihit consits of one conv layer and triple hit layers
def trihit_cth(num_classes=100, dp_rate=0.0):
    model = trihit_x(num_classes=num_classes, depths=[2, 2, 4, 2], dims=[32, 64, 128, 256], arch=['conv', 'hit', 'hit', 'hit'], drop_path_rate=dp_rate)
    return model
